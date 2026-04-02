import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from fusion.base import FusionModule
from building_blocks import StructureEncoder

# ---------------------------------------------------------------------------
# CMD (Central Moment Discrepancy) — used for similarity loss
# Reference: Zellinger et al., "Central Moment Discrepancy (CMD) for
# Domain-Invariant Representation Learning", ICLR 2017
# ---------------------------------------------------------------------------

def _cmd_k(x1: torch.Tensor, x2: torch.Tensor, K: int = 5) -> torch.Tensor:
    """
    Compute Central Moment Discrepancy between two batched feature matrices.

    Parameters
    ----------
    x1 : [B, d]
    x2 : [B, d]
    K  : number of moment orders to match (default 5, as in MISA paper)

    Returns
    -------
    cmd : scalar Tensor
    """
    # First order: difference of means
    mu1 = x1.mean(dim=0)
    mu2 = x2.mean(dim=0)
    cmd = (mu1 - mu2).norm(p=2)

    # Higher order central moments
    cx1 = x1 - mu1.unsqueeze(0)
    cx2 = x2 - mu2.unsqueeze(0)

    for k in range(2, K + 1):
        # k-th central moment: E[(x - mu)^k] computed element-wise then L2-normed
        m1 = cx1.pow(k).mean(dim=0)
        m2 = cx2.pow(k).mean(dim=0)
        cmd = cmd + (m1 - m2).norm(p=2)

    return cmd


# ---------------------------------------------------------------------------
# Helper: zero-mean + unit L2 norm (for difference loss preprocessing)
# ---------------------------------------------------------------------------

def _normalize_batch(H: torch.Tensor) -> torch.Tensor:
    """
    Center to zero mean and normalize to unit L2 norm along the feature dim.

    Parameters
    ----------
    H : [B, d] batched hidden vectors

    Returns
    -------
    H_norm : [B, d]
    """
    H = H - H.mean(dim=0, keepdim=True)
    norms = H.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    return H / norms


# ---------------------------------------------------------------------------
# MISAFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class MISAFusion(FusionModule):
    """
    MISA (Modality-Invariant and -Specific Representations) fusion module.

    Architecture:
        - One shared invariant encoder E_c (parameters shared across modalities)
        - Three separate specific encoders E_p (one per modality)
        - Transformer self-attention over the 6 resulting vectors
        - Reconstruction decoder D to recover original u_m from h_m^c + h_m^p

    Loss (returned by compute_pretrain_loss):
        L = alpha * L_sim + beta * L_diff + gamma * L_recon

    Parameters
    ----------
    input_dims   : list of input dims per modality, e.g. [512, 512, 512]
    r            : projection dimension (d_h in the paper)
    n_heads      : number of attention heads in the Transformer layer
    cmd_K        : number of moment orders for CMD (default 5)
    alpha        : weight for similarity loss
    beta         : weight for difference loss
    gamma        : weight for reconstruction loss
    """

    def __init__(
        self,
        input_dims: List[int],
        r: int,
        hidden_dims: List[int] = [256, 128],
        n_heads: int = 4,
        cmd_K: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        # out_dim = 6 * r (3 invariant + 3 specific, each of dim r)
        super().__init__(out_dim=6 * r, has_pretrain=True)

        self.r       = r
        self.n_heads = n_heads
        self.cmd_K   = cmd_K
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma

        # --- Shared invariant encoder E_c ---
        # Per-modality input projections to unify dims before shared body
        self.invariant_input_projs = nn.ModuleList([
            nn.Linear(input_dims[m], r) for m in range(3)
        ])
        self.invariant_body = StructureEncoder(
            input_dim=r,
            hidden_dims=hidden_dims,
            output_dim=r,
        )

        # --- Separate specific encoders E_p (one StructureEncoder per modality) ---
        self.specific_encoders = nn.ModuleList([
            StructureEncoder(
                input_dim=input_dims[m],
                hidden_dims=hidden_dims,
                output_dim=r,
            )
            for m in range(3)
        ])

        # --- Transformer self-attention over the 6 vectors ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=r,
            nhead=n_heads,
            dim_feedforward=r * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # --- Reconstruction decoders (per-modality) ---
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(r, r),
                nn.ReLU(),
                nn.Linear(r, input_dims[m]),
            )
            for m in range(3)
        ])

    def _encode(
        self, x_list: List[torch.Tensor]
    ) -> tuple:
        """
        Encode each modality into invariant and specific representations.

        Parameters
        ----------
        x_list : list of [B, d_m] tensors (one per modality)

        Returns
        -------
        h_c : list of [B, r] invariant representations
        h_p : list of [B, r] specific representations
        """
        h_c = []
        h_p = []
        for m in range(3):
            # Invariant: per-modality input proj -> shared StructureEncoder body
            proj_m = self.invariant_input_projs[m](x_list[m])
            h_c.append(self.invariant_body(proj_m))

            # Specific: separate StructureEncoder per modality
            h_p.append(self.specific_encoders[m](x_list[m]))

        return h_c, h_p

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality encodings via MISA.

        Steps:
            1. Encode into invariant (h_c) and specific (h_p) representations
            2. Stack 6 vectors into [B, 6, r] matrix
            3. Pass through Transformer self-attention
            4. Flatten to [B, 6*r]

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        [B, 6*r] tensor
        """
        h_c, h_p = self._encode(x_list)

        # Stack: [B, 6, r] — order: h_l^c, h_v^c, h_a^c, h_l^p, h_v^p, h_a^p
        M = torch.stack(h_c + h_p, dim=1)  # [B, 6, r]

        # Transformer self-attention
        M_tilde = self.transformer(M)       # [B, 6, r]

        # Flatten to joint vector
        return M_tilde.reshape(M_tilde.size(0), -1)  # [B, 6*r]

    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        **kwargs,
    ) -> dict:
        """
        Compute MISA regularization losses.

        L = alpha * L_sim + beta * L_diff + gamma * L_recon

        Parameters
        ----------
        x_list : list of [B, d_m] tensors (original encoder outputs u_m)

        Returns
        -------
        dict:
            total  : weighted sum of all three losses (scalar Tensor)
            sim    : similarity loss value (float)
            diff   : difference loss value (float)
            recon  : reconstruction loss value (float)
        """
        h_c, h_p = self._encode(x_list)

        # ----- Similarity Loss (CMD between all pairs of invariant reprs) -----
        pairs = [(0, 1), (0, 2), (1, 2)]
        loss_sim = torch.tensor(0.0, device=x_list[0].device)
        for m1, m2 in pairs:
            loss_sim = loss_sim + _cmd_k(h_c[m1], h_c[m2], K=self.cmd_K)
        loss_sim = loss_sim / 3.0

        # ----- Difference Loss (soft orthogonality) -----
        # Normalize to zero mean + unit L2 norm before computing
        h_c_norm = [_normalize_batch(h) for h in h_c]
        h_p_norm = [_normalize_batch(h) for h in h_p]

        loss_diff = torch.tensor(0.0, device=x_list[0].device)

        # Term 1: ||H_m^c^T H_m^p||_F^2 for each modality
        for m in range(3):
            cross = h_c_norm[m].T @ h_p_norm[m]  # [r, r]
            loss_diff = loss_diff + cross.norm(p="fro").pow(2)

        # Term 2: ||H_m1^p^T H_m2^p||_F^2 for each pair
        for m1, m2 in pairs:
            cross = h_p_norm[m1].T @ h_p_norm[m2]  # [r, r]
            loss_diff = loss_diff + cross.norm(p="fro").pow(2)

        # ----- Reconstruction Loss (MSE between decoded and original u_m) -----
        loss_recon = torch.tensor(0.0, device=x_list[0].device)
        for m in range(3):
            u_hat = self.decoders[m](h_c[m] + h_p[m])  # [B, d_m]
            d_m = x_list[m].size(1)
            loss_recon = loss_recon + F.mse_loss(u_hat, x_list[m]) * d_m / d_m
            # Paper formula: ||u_m - u_hat||_2^2 / d_h, MSE already averages
        loss_recon = loss_recon / 3.0

        # ----- Total -----
        total = (
            self.alpha * loss_sim
            + self.beta * loss_diff
            + self.gamma * loss_recon
        )

        return {
            "total": total,
            "sim"  : loss_sim.item(),
            "diff" : loss_diff.item(),
            "recon": loss_recon.item(),
        }