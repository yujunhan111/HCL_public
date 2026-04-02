import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from fusion.base import FusionModule
from building_blocks import StructureEncoder

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _FFNEncoder(nn.Module):
    """
    Shared or specific encoder implemented as a small Transformer encoder.
    Input is [B, hidden_size] -> unsqueeze to [B, 1, hidden_size] ->
    project to d -> Transformer layers -> squeeze back to [B, d].
    """

    def __init__(self, input_dim: int, d: int, n_layers: int = 1, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, input_dim]
        returns : [B, d]
        """
        h = self.input_proj(x).unsqueeze(1)   # [B, 1, d]
        h = self.transformer(h)                # [B, 1, d]
        return h.squeeze(1)                    # [B, d]


class _ConvDecoder(nn.Module):
    """
    1D convolution decoder: reconstruct original input from
    concatenated shared + specific features.
    [B, 2*d] -> [B, input_dim]
    """

    def __init__(self, d: int, output_dim: int):
        super().__init__()
        # Use Conv1d: treat feature dim as "length", single channel
        self.net = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, output_dim),
        )

    def forward(self, sh: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """
        sh : [B, d]  shared features
        sp : [B, d]  specific features
        returns : [B, output_dim]
        """
        return self.net(torch.cat([sh, sp], dim=-1))


class _MultimodalCrossAttention(nn.Module):
    """
    Language-Focused Cross-Attention block.
    Q comes from language, K/V come from another modality.
    Input/output are [B, d] vectors (unsqueezed to length-1 sequences internally).
    """

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.ReLU(),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q  : [B, d]  language query
        kv : [B, d]  key-value source modality
        returns : [B, d]
        """
        # Unsqueeze to sequence length 1 for MultiheadAttention
        q_seq  = q.unsqueeze(1)    # [B, 1, d]
        kv_seq = kv.unsqueeze(1)   # [B, 1, d]

        attn_out, _ = self.cross_attn(q_seq, kv_seq, kv_seq)  # [B, 1, d]
        h = self.norm1(q_seq + self.dropout(attn_out))         # [B, 1, d]
        h = self.norm2(h + self.ffn(h))                        # [B, 1, d]
        return h.squeeze(1)                                    # [B, d]


# ---------------------------------------------------------------------------
# Triplet mining helpers
# ---------------------------------------------------------------------------

def _mine_triplets(
    sh_list: List[torch.Tensor],
    labels: torch.Tensor,
) -> Optional[tuple]:
    """
    Mine triplets for modified triplet loss in the shared space.

    For each anchor (modality m, sample i):
        Positive: same label, different modality
        Negative: different label, same modality

    Parameters
    ----------
    sh_list : [sh_L, sh_V, sh_A], each [B, d]
    labels  : [B] integer labels

    Returns
    -------
    (anchors, positives, negatives) each [num_triplets, d], or None if no valid triplets
    """
    B = labels.size(0)
    device = labels.device

    if B < 2:
        return None

    anchors, positives, negatives = [], [], []

    for m in range(3):
        for i in range(B):
            # Positive: same label, different modality
            pos_mod_candidates = [j for j in range(3) if j != m]
            pos_found = False
            for pm in pos_mod_candidates:
                # Any sample with same label in another modality
                same_label_mask = (labels == labels[i])
                if same_label_mask.any():
                    # Pick the first matching sample
                    idx = same_label_mask.nonzero(as_tuple=True)[0][0].item()
                    pos_vec = sh_list[pm][idx]
                    pos_found = True
                    break

            if not pos_found:
                continue

            # Negative: different label, same modality
            diff_label_mask = (labels != labels[i])
            if not diff_label_mask.any():
                continue
            neg_idx = diff_label_mask.nonzero(as_tuple=True)[0][0].item()
            neg_vec = sh_list[m][neg_idx]

            anchors.append(sh_list[m][i])
            positives.append(pos_vec)
            negatives.append(neg_vec)

    if len(anchors) == 0:
        return None

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
    )


# ---------------------------------------------------------------------------
# DLFFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class DLFFusion(FusionModule):
    """
    DLF (Disentangled-Language-Focused) fusion module.

    Architecture:
        1. Feature Disentanglement Module (FDM):
           - Shared encoder (parameters shared across modalities)
           - Specific encoders (one per modality)
           - Conv decoders for reconstruction
        2. Language-Focused Attractor (LFA):
           - Cross-attention: Language query attends to L/V/A specific features
        3. Higher-level projection:
           - Shared features -> unified Transformer -> HSh
           - Enhanced specific features -> projection -> HSp^m
        4. Fusion: Concat(HSp^L, HSp^V, HSp^A, HSh) -> [B, 4*r]

    Pretrain loss (L_d):
        L_r (reconstruction) + L_s (specific consistency) +
        L_m (modified triplet) + L_o (soft orthogonality)

    Parameters
    ----------
    input_dims   : list of input dims per modality, e.g. [512, 512, 512]
    r            : latent dimension (shared across all internal projections)
    n_layers     : number of Transformer layers in encoders
    n_heads      : number of attention heads
    dropout      : dropout rate
    triplet_margin : margin for modified triplet loss
    lambda_r     : weight for reconstruction loss
    lambda_s     : weight for specific consistency loss
    lambda_m     : weight for modified triplet loss
    lambda_o     : weight for soft orthogonality loss
    """
    def __init__(
        self,
        input_dims: List[int],
        r: int,
        hidden_dims: List[int] = [256, 128],
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        triplet_margin: float = 1.0,
        lambda_r: float = 1.0,
        lambda_s: float = 1.0,
        lambda_m: float = 1.0,
        lambda_o: float = 1.0,
    ):
        # out_dim = 4 * r (HSp^L + HSp^V + HSp^A + HSh)
        super().__init__(out_dim=4 * r, has_pretrain=True)

        self.r = r
        self.n_heads = n_heads
        self.triplet_margin = triplet_margin
        self.lambda_r = lambda_r
        self.lambda_s = lambda_s
        self.lambda_m = lambda_m
        self.lambda_o = lambda_o

        # --- FDM: Shared encoder (one StructureEncoder, applied to all modalities) ---
        # Per-modality input projections to unify input dims before shared encoder
        self.shared_input_projs = nn.ModuleList([
            nn.Linear(input_dims[m], r) for m in range(3)
        ])
        self.shared_body = StructureEncoder(
            input_dim=r,
            hidden_dims=hidden_dims,
            output_dim=r,
        )

        # --- FDM: Specific encoders (one StructureEncoder per modality) ---
        self.specific_encoders = nn.ModuleList([
            StructureEncoder(
                input_dim=input_dims[m],
                hidden_dims=hidden_dims,
                output_dim=r,
            )
            for m in range(3)
        ])

        # --- FDM: Reconstruction decoders (one per modality) ---
        self.decoders = nn.ModuleList([
            _ConvDecoder(r, input_dims[m]) for m in range(3)
        ])

        # --- LFA: Language-Focused Cross-Attention (3 branches) ---
        self.lfa_branches = nn.ModuleList([
            _MultimodalCrossAttention(r, n_heads=n_heads, dropout=dropout)
            for _ in range(3)
        ])

        # --- Higher-level projections ---
        # Shared: unified Transformer + FC -> HSh
        hsh_layer = nn.TransformerEncoderLayer(
            d_model=r,
            nhead=n_heads,
            dim_feedforward=r * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.hsh_transformer = nn.TransformerEncoder(hsh_layer, num_layers=1)
        self.hsh_fc = nn.Linear(r, r)

        # Specific: projection to higher-level specific features HSp^m
        self.hsp_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(r, r), nn.ReLU()) for _ in range(3)
        ])

    def _encode_shared(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Encode each modality through the shared encoder.

        Parameters
        ----------
        x_list : [x_L, x_V, x_A], each [B, input_dim_m]

        Returns
        -------
        sh_list : [sh_L, sh_V, sh_A], each [B, r]
        """
        sh_list = []
        for m in range(3):
            proj = self.shared_input_projs[m](x_list[m])  # [B, r]
            sh_list.append(self.shared_body(proj))          # [B, r]
        return sh_list
    def _encode_specific(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Encode each modality through its specific StructureEncoder.

        Parameters
        ----------
        x_list : [x_L, x_V, x_A], each [B, input_dim_m]

        Returns
        -------
        sp_list : [sp_L, sp_V, sp_A], each [B, r]
        """
        return [self.specific_encoders[m](x_list[m]) for m in range(3)]

    def _reconstruct(
        self,
        sh_list: List[torch.Tensor],
        sp_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Reconstruct original inputs from shared + specific features.

        Returns
        -------
        x_recon : [x'_L, x'_V, x'_A], each [B, input_dim_m]
        """
        return [self.decoders[m](sh_list[m], sp_list[m]) for m in range(3)]

    def _run_lfa(self, sp_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Language-Focused Attractor: language query attends to each modality.

        Parameters
        ----------
        sp_list : [sp_L, sp_V, sp_A], each [B, r]

        Returns
        -------
        enhanced : [enh_L, enh_V, enh_A], each [B, r]
            enh_m = LFA(Q=sp_L, KV=sp_m)
        """
        q_lang = sp_list[0]  # Language specific features as query
        enhanced = []
        for m in range(3):
            enh = self.lfa_branches[m](q_lang, sp_list[m])  # [B, r]
            enhanced.append(enh)
        return enhanced

    def _compute_hsh(self, sh_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute higher-level shared features HSh.
        Average shared features across modalities, then Transformer + FC.

        Parameters
        ----------
        sh_list : [sh_L, sh_V, sh_A], each [B, r]

        Returns
        -------
        hsh : [B, r]
        """
        # Stack as sequence of 3 tokens: [B, 3, r]
        sh_seq = torch.stack(sh_list, dim=1)           # [B, 3, r]
        h = self.hsh_transformer(sh_seq)               # [B, 3, r]
        # Mean pool over the 3 modality tokens
        h = h.mean(dim=1)                              # [B, r]
        return self.hsh_fc(h)                          # [B, r]

    def _compute_hsp(self, enhanced: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Project enhanced specific features to higher-level specific features.

        Parameters
        ----------
        enhanced : [enh_L, enh_V, enh_A], each [B, r]

        Returns
        -------
        hsp_list : [HSp_L, HSp_V, HSp_A], each [B, r]
        """
        return [self.hsp_projections[m](enhanced[m]) for m in range(3)]

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Full forward pass: FDM -> LFA -> Fusion.

        Parameters
        ----------
        x_list : [x_L, x_V, x_A], each [B, input_dim_m]

        Returns
        -------
        fused : [B, 4*r]
        """
        # FDM
        sh_list = self._encode_shared(x_list)
        sp_list = self._encode_specific(x_list)

        # LFA
        enhanced = self._run_lfa(sp_list)

        # Higher-level features
        hsh      = self._compute_hsh(sh_list)     # [B, r]
        hsp_list = self._compute_hsp(enhanced)     # 3 x [B, r]

        # Fusion: Concat(HSp^L, HSp^V, HSp^A, HSh)
        return torch.cat(hsp_list + [hsh], dim=-1)  # [B, 4*r]

    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Compute the decoupling loss L_d = lambda_r * L_r + lambda_s * L_s
                                        + lambda_m * L_m + lambda_o * L_o

        Parameters
        ----------
        x_list : [x_L, x_V, x_A], each [B, input_dim_m]
        labels : [B] integer labels, needed for triplet mining.
                 If None, triplet loss is skipped.

        Returns
        -------
        dict with keys: total, recon, specific, triplet, ortho
        """
        device = x_list[0].device

        # --- FDM encode ---
        sh_list = self._encode_shared(x_list)
        sp_list = self._encode_specific(x_list)

        # --- L_r: Reconstruction loss ---
        x_recon = self._reconstruct(sh_list, sp_list)
        loss_r = torch.tensor(0.0, device=device)
        for m in range(3):
            loss_r = loss_r + F.mse_loss(x_recon[m], x_list[m])
        loss_r = loss_r / 3.0

        # --- L_s: Specific consistency loss ---
        # Pass reconstructed input back through specific encoder
        loss_s = torch.tensor(0.0, device=device)
        for m in range(3):
            sp_prime = self.specific_encoders[m](x_recon[m].detach())
            loss_s = loss_s + F.mse_loss(sp_prime, sp_list[m].detach())
        loss_s = loss_s / 3.0

        # --- L_m: Modified triplet loss ---
        loss_m = torch.tensor(0.0, device=device)
        if labels is not None:
            triplets = _mine_triplets(sh_list, labels)
            if triplets is not None:
                anchors, positives, negatives = triplets
                # Cosine similarity (higher = more similar)
                pos_sim = F.cosine_similarity(anchors, positives, dim=-1)
                neg_sim = F.cosine_similarity(anchors, negatives, dim=-1)
                # Triplet: want pos_sim > neg_sim + margin
                # loss = max(0, neg_sim - pos_sim + margin)
                loss_m = torch.clamp(neg_sim - pos_sim + self.triplet_margin, min=0.0).mean()

        # --- L_o: Soft orthogonality loss ---
        # Non-negative cosine similarity between shared and specific
        loss_o = torch.tensor(0.0, device=device)
        for m in range(3):
            cos_sim = F.cosine_similarity(sh_list[m], sp_list[m], dim=-1)
            loss_o = loss_o + torch.clamp(cos_sim, min=0.0).mean()
        loss_o = loss_o / 3.0

        # --- Total decoupling loss ---
        total = (
            self.lambda_r * loss_r
            + self.lambda_s * loss_s
            + self.lambda_m * loss_m
            + self.lambda_o * loss_o
        )

        return {
            "total"   : total,
            "recon"   : loss_r.item(),
            "specific": loss_s.item(),
            "triplet" : loss_m.item(),
            "ortho"   : loss_o.item(),
        }