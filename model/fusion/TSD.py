import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import List, Optional

from fusion.base import FusionModule


# ---------------------------------------------------------------------------
# Number of modalities and pair indices
# ---------------------------------------------------------------------------
_NUM_MODALITIES = 3
_MODALITY_PAIRS = [(0, 1), (0, 2), (1, 2)]  # (l,v), (l,a), (v,a)
_PAIR_INDEX = {(0, 1): 0, (0, 2): 1, (1, 2): 2}


# ---------------------------------------------------------------------------
# HSIC (Hilbert-Schmidt Independence Criterion) with RBF kernel
# ---------------------------------------------------------------------------

def _rbf_kernel(X: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute RBF kernel matrix K[i,j] = exp(-||x_i - x_j||^2 / (2 * sigma^2)).

    Parameters
    ----------
    X : [B, d]
    sigma : kernel bandwidth

    Returns
    -------
    K : [B, B]
    """
    dist_sq = torch.cdist(X, X, p=2).pow(2)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def _hsic(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute biased HSIC estimator between two batched feature matrices.

    Parameters
    ----------
    X : [B, d1]
    Y : [B, d2]
    sigma : RBF kernel bandwidth

    Returns
    -------
    hsic : scalar Tensor
    """
    B = X.size(0)
    if B < 4:
        return torch.tensor(0.0, device=X.device)

    K = _rbf_kernel(X, sigma)
    L = _rbf_kernel(Y, sigma)

    # Centering matrix H = I - (1/B) * 11^T
    H = torch.eye(B, device=X.device) - 1.0 / B
    # HSIC = (1 / (B-1)^2) * tr(KHLH)
    KH = K @ H
    LH = L @ H
    hsic = (KH * LH.T).sum() / ((B - 1) ** 2)
    return hsic


# ---------------------------------------------------------------------------
# Decoupling Supervisor: 3-branch classifier
# ---------------------------------------------------------------------------

class _DecouplingSupervisor(nn.Module):
    """
    Three-branch MLP classifier that predicts the source subspace
    (common / submodally-shared / private) of pooled embeddings.

    Each branch: Linear -> GELU -> Linear -> 1 logit (sigmoid probability).
    """

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.D_com = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.D_sub = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.D_pri = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        c_list: List[torch.Tensor],
        s_list: List[List[torch.Tensor]],
        p_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute decoupling supervisor loss L_sup.

        Parameters
        ----------
        c_list : [c_0, c_1, c_2], each [B, d_c] — common features per modality
        s_list : s_list[m] = list of s_{mn}^{(m)} for each n != m, each [B, d_c]
        p_list : [p_0, p_1, p_2], each [B, d_c] — private features per modality

        Returns
        -------
        loss : scalar Tensor
        """
        eps = 1e-8
        loss = torch.tensor(0.0, device=c_list[0].device)

        for m in range(_NUM_MODALITIES):
            # Common branch: D_com should output high probability for c_m
            log_d_com = torch.log(torch.sigmoid(self.D_com(c_list[m])) + eps).mean()
            loss = loss + log_d_com

            # Submodally-shared branch: D_sub should output high prob for s_{mn}^{(m)}
            for s_mn_m in s_list[m]:
                log_d_sub = torch.log(torch.sigmoid(self.D_sub(s_mn_m)) + eps).mean()
                loss = loss + log_d_sub

            # Private branch: D_pri should output high probability for p_m
            log_d_pri = torch.log(torch.sigmoid(self.D_pri(p_list[m])) + eps).mean()
            loss = loss + log_d_pri

        # Negative because we maximize log-likelihood => minimize -log
        loss = -loss / _NUM_MODALITIES
        return loss


# ---------------------------------------------------------------------------
# TSDFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class TSDFusion(FusionModule):
    """
    Tri-Subspaces Disentanglement (TSD) fusion module.

    Architecture:
        1. Projection: per-modality Linear -> d_z
        2. Common Encoder (shared): 2 FC + GELU + LayerNorm -> d_c
        3. Submodally-Shared Encoders (one per pair, shared params): FC + Sigmoid -> d_c
        4. Private Encoders (one per modality): FC + Sigmoid -> d_c
        5. Hierarchical Gated Fusion: 9 subspace features -> adaptive weighted sum -> d_c

    Loss (compute_pretrain_loss):
        L_com + lambda_1 * L_pair + lambda_2 * L_pri + lambda_3 * L_ort + lambda_4 * L_sup

    Parameters
    ----------
    input_dims   : list of input dims per modality, e.g. [512, 512, 512]
    r            : output dimension (d_c = d_z = r)
    lambda_pair  : weight for pairwise collaboration loss
    lambda_pri   : weight for private disparity loss (HSIC)
    lambda_ort   : weight for orthogonality loss
    lambda_sup   : weight for decoupling supervisor loss
    hsic_sigma   : RBF kernel bandwidth for HSIC
    """

    def __init__(
        self,
        input_dims: List[int],
        r: int,
        lambda_pair: float = 1.0,
        lambda_pri: float = 1.0,
        lambda_ort: float = 1.0,
        lambda_sup: float = 1.0,
        hsic_sigma: float = 1.0,
    ):
        super().__init__(out_dim=r, has_pretrain=True)

        self.r = r
        self.lambda_pair = lambda_pair
        self.lambda_pri = lambda_pri
        self.lambda_ort = lambda_ort
        self.lambda_sup = lambda_sup
        self.hsic_sigma = hsic_sigma

        # --- Projection layers: input_dim -> r per modality ---
        self.projections = nn.ModuleList([
            nn.Linear(input_dims[m], r) for m in range(_NUM_MODALITIES)
        ])

        # --- Common Encoder (shared across all modalities) ---
        # 2 FC layers with GELU and LayerNorm
        self.common_encoder = nn.Sequential(
            nn.Linear(r, r),
            nn.GELU(),
            nn.LayerNorm(r),
            nn.Linear(r, r),
            nn.GELU(),
            nn.LayerNorm(r),
        )

        # --- Submodally-Shared Encoders (one per pair, shared params within pair) ---
        # 3 pairs: (0,1), (0,2), (1,2)
        self.sub_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(r, r), nn.Sigmoid())
            for _ in range(len(_MODALITY_PAIRS))
        ])

        # --- Private Encoders (one per modality) ---
        self.private_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(r, r), nn.Sigmoid())
            for _ in range(_NUM_MODALITIES)
        ])

        # --- Hierarchical Gated Fusion ---
        # 9 subspace features: 3 common + 3 sub-shared + 3 private
        self.num_subspaces = 9
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(r * self.num_subspaces, r),
                nn.ReLU(),
                nn.Linear(r, 1),
            )
            for _ in range(self.num_subspaces)
        ])

        # --- Decoupling Supervisor ---
        self.supervisor = _DecouplingSupervisor(d=r)

    def _encode_subspaces(
        self, x_list: List[torch.Tensor]
    ) -> tuple:
        """
        Project inputs and encode into three subspaces.

        Parameters
        ----------
        x_list : [x_0, x_1, x_2], each [B, input_dim_m]

        Returns
        -------
        c_list : [c_0, c_1, c_2], each [B, r] — common features
        s_dict : dict (m, n) -> (s_mn_m, s_mn_n), each [B, r] — sub-shared features
        p_list : [p_0, p_1, p_2], each [B, r] — private features
        z_list : [z_0, z_1, z_2], each [B, r] — projected features (for reference)
        """
        # Project to unified space
        z_list = [self.projections[m](x_list[m]) for m in range(_NUM_MODALITIES)]

        # Common encoder (shared params)
        c_list = [self.common_encoder(z_list[m]) for m in range(_NUM_MODALITIES)]

        # Submodally-shared encoders
        s_dict = {}
        for pair_idx, (m, n) in enumerate(_MODALITY_PAIRS):
            encoder = self.sub_encoders[pair_idx]
            s_dict[(m, n)] = (encoder(z_list[m]), encoder(z_list[n]))

        # Private encoders
        p_list = [self.private_encoders[m](z_list[m]) for m in range(_NUM_MODALITIES)]

        return c_list, s_dict, p_list, z_list

    def _gated_fusion(
        self,
        c_list: List[torch.Tensor],
        s_dict: dict,
        p_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Hierarchical gated fusion of all 9 subspace features.

        Collect: [c_0, c_1, c_2, s_01, s_02, s_12, p_0, p_1, p_2]
        where s_mn = mean(s_mn^{(m)}, s_mn^{(n)}).

        Parameters
        ----------
        c_list : 3 x [B, r]
        s_dict : 3 pairs -> (s_m, s_n), each [B, r]
        p_list : 3 x [B, r]

        Returns
        -------
        y_final : [B, r]
        """
        # Average the two outputs of each sub-shared encoder for fusion
        s_fused = []
        for m, n in _MODALITY_PAIRS:
            s_m, s_n = s_dict[(m, n)]
            s_fused.append((s_m + s_n) / 2.0)

        # Collect all 9 features: 3 common + 3 sub-shared + 3 private
        all_features = c_list + s_fused + p_list  # list of 9 x [B, r]

        # Concatenate for gate input
        gate_input = torch.cat(all_features, dim=-1)  # [B, 9*r]

        # Compute gate logits and apply softmax
        gate_logits = torch.stack([
            self.gate_networks[k](gate_input).squeeze(-1)
            for k in range(self.num_subspaces)
        ], dim=-1)  # [B, 9]

        weights = F.softmax(gate_logits, dim=-1)  # [B, 9]

        # Weighted sum
        features_stacked = torch.stack(all_features, dim=1)  # [B, 9, r]
        y_final = (weights.unsqueeze(-1) * features_stacked).sum(dim=1)  # [B, r]

        return y_final

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: encode subspaces and fuse.

        Parameters
        ----------
        x_list : [x_0, x_1, x_2], each [B, input_dim_m]

        Returns
        -------
        y : [B, r]
        """
        c_list, s_dict, p_list, _ = self._encode_subspaces(x_list)
        return self._gated_fusion(c_list, s_dict, p_list)

    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        **kwargs,
    ) -> dict:
        """
        Compute TSD regularization losses.

        L = L_com + lambda_1 * L_pair + lambda_2 * L_pri
            + lambda_3 * L_ort + lambda_4 * L_sup

        Parameters
        ----------
        x_list : [x_0, x_1, x_2], each [B, input_dim_m]

        Returns
        -------
        dict with keys: total, com, pair, pri, ort, sup
        """
        device = x_list[0].device
        c_list, s_dict, p_list, _ = self._encode_subspaces(x_list)

        # ===== L_com: Common Consistency Loss =====
        # Mean squared distance between all pairs of common features
        loss_com = torch.tensor(0.0, device=device)
        for m, n in _MODALITY_PAIRS:
            loss_com = loss_com + (c_list[m] - c_list[n]).pow(2).mean()
        loss_com = loss_com / len(_MODALITY_PAIRS)

        # ===== L_pair: Pairwise Collaboration Loss =====
        # Mean squared distance between paired sub-shared features
        loss_pair = torch.tensor(0.0, device=device)
        for m, n in _MODALITY_PAIRS:
            s_m, s_n = s_dict[(m, n)]
            loss_pair = loss_pair + (s_m - s_n).pow(2).mean()
        loss_pair = loss_pair / len(_MODALITY_PAIRS)

        # ===== L_pri: Private Disparity Loss (HSIC) =====
        # Encourage independence between private features of different modalities
        loss_pri = torch.tensor(0.0, device=device)
        for m, n in _MODALITY_PAIRS:
            loss_pri = loss_pri + _hsic(p_list[m], p_list[n], sigma=self.hsic_sigma)
        loss_pri = loss_pri / len(_MODALITY_PAIRS)

        # ===== L_ort: Orthogonality Loss =====
        # Penalize overlap between subspaces using Frobenius norm
        loss_ort = torch.tensor(0.0, device=device)
        for m in range(_NUM_MODALITIES):
            # C_m^T @ P_m
            loss_ort = loss_ort + (c_list[m].T @ p_list[m]).pow(2).sum()

            for n in range(_NUM_MODALITIES):
                if n == m:
                    continue
                # Find the pair key
                pair_key = (min(m, n), max(m, n))
                s_m = s_dict[pair_key][0] if pair_key[0] == m else s_dict[pair_key][1]

                # S_{mn}^{(m)T} @ P_m
                loss_ort = loss_ort + (s_m.T @ p_list[m]).pow(2).sum()
                # S_{mn}^{(m)T} @ C_m
                loss_ort = loss_ort + (s_m.T @ c_list[m]).pow(2).sum()

        loss_ort = loss_ort / _NUM_MODALITIES

        # ===== L_sup: Decoupling Supervisor Loss =====
        # Build s_list[m] = list of s_{mn}^{(m)} for n != m
        s_list_per_mod = [[] for _ in range(_NUM_MODALITIES)]
        for m, n in _MODALITY_PAIRS:
            s_m, s_n = s_dict[(m, n)]
            s_list_per_mod[m].append(s_m)
            s_list_per_mod[n].append(s_n)

        loss_sup = self.supervisor(c_list, s_list_per_mod, p_list)

        # ===== Total =====
        total = (
            loss_com
            + self.lambda_pair * loss_pair
            + self.lambda_pri * loss_pri
            + self.lambda_ort * loss_ort
            + self.lambda_sup * loss_sup
        )

        return {
            "total": total,
            "com":   loss_com.item(),
            "pair":  loss_pair.item(),
            "pri":   loss_pri.item(),
            "ort":   loss_ort.item(),
            "sup":   loss_sup.item(),
        }