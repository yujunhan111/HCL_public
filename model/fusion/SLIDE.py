import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import List, Optional

from fusion.base import FusionModule


# ---------------------------------------------------------------------------
# Structure table helpers (module-level, built once at import time)
# Mirrors the same logic used in HCLFusion so structures are aligned.
# ---------------------------------------------------------------------------

_NUM_MODALITIES = 3
_NUM_STRUCTURES = 7  # 2^3 - 1


def _build_structure_tables(M: int):
    """
    Build lookup tables for M modalities.

    Returns
    -------
    structure_list : list of frozenset
        All 2^M - 1 non-empty subsets ordered by descending size then
        ascending lex order.
    structure_mask : dict[int, dict[int, int]]
        structure_mask[m][k] = 1 means modality m participates in structure k.
    """
    subsets = []
    for size in range(M, 0, -1):
        for combo in combinations(range(M), size):
            subsets.append(frozenset(combo))

    # mask[m][k] = 1  iff modality m belongs to structure k
    mask = {m: {} for m in range(M)}
    for k, subset in enumerate(subsets):
        for m in subset:
            mask[m][k] = 1

    return subsets, mask


_STRUCTURE_LIST, _STRUCTURE_MASK = _build_structure_tables(_NUM_MODALITIES)


def _build_S_matrix(r: int, device: torch.device) -> torch.Tensor:
    """
    Build the fixed binary structure matrix S of shape [d, total_r].

    Each of the 7 structures gets exactly r columns.
    S[i, k*r : (k+1)*r] = 1  iff modality i participates in structure k.

    Returns
    -------
    S : torch.Tensor, shape [_NUM_MODALITIES, _NUM_STRUCTURES * r], dtype float32
    """
    total_r = _NUM_STRUCTURES * r
    S = torch.zeros(_NUM_MODALITIES, total_r, device=device)
    for k in range(_NUM_STRUCTURES):
        for m in _STRUCTURE_MASK:
            if k in _STRUCTURE_MASK[m]:
                S[m, k * r : (k + 1) * r] = 1.0
    return S


# ---------------------------------------------------------------------------
# SLIDE iterative solver (Algorithm 2 from the paper)
# ---------------------------------------------------------------------------
def _slide_fit(
    X_list: List[torch.Tensor],
    S: torch.Tensor,
    total_r: int,
    max_iter: int,
    tol: float,
) -> tuple:
    """
    Fit SLIDE with a pre-specified binary structure matrix S (Algorithm 2).
    Runs on full data (no batching). Shows iteration progress via tqdm.

    Returns
    -------
    U : torch.Tensor, shape [B, total_r] — score matrix
    V : torch.Tensor, shape [p, total_r] — loading matrix (save for out-of-sample projection)
    """
    from tqdm import tqdm

    device = X_list[0].device
    dtype  = X_list[0].dtype
    B      = X_list[0].size(0)

    X = torch.cat(X_list, dim=1)  # [B, p]
    p = X.size(1)

    boundaries = []
    start = 0
    for Xi in X_list:
        end = start + Xi.size(1)
        boundaries.append((start, end))
        start = end

    # Initialize U via SVD of X
    k_svd = min(total_r, B, p)
    try:
        Ur, _, _ = torch.linalg.svd(X, full_matrices=False)  # [B, min(B,p)]
        U = Ur[:, :k_svd]
        if k_svd < total_r:
            extra = torch.randn(B, total_r - k_svd, device=device, dtype=dtype)
            extra = torch.linalg.qr(extra)[0]
            U = torch.cat([U, extra], dim=1)
    except Exception:
        U = torch.linalg.qr(torch.randn(B, total_r, device=device, dtype=dtype))[0]

    UV_prev = torch.zeros(B, p, device=device, dtype=dtype)

    pbar = tqdm(range(max_iter), desc="SLIDE iterations", leave=True)
    for it in pbar:
        # Update V block by block
        V = torch.zeros(p, total_r, device=device, dtype=dtype)
        for m, (s, e) in enumerate(boundaries):
            Xi    = X_list[m]                  # [B, hidden_i]
            s_row = S[m]                        # [total_r] binary mask
            active = s_row.bool()

            if active.any():
                U_active       = U[:, active]  # [B, num_active]
                V[s:e, active] = Xi.T @ U_active

        # Update U via SVD of X @ V
        XV = X @ V
        try:
            R, _, Qt = torch.linalg.svd(XV, full_matrices=False)
            k = min(B, total_r)
            U = R[:, :k] @ Qt[:k, :]
        except Exception:
            U, _ = torch.linalg.qr(XV)
            U = U[:, :total_r]

        # Check convergence
        UV_curr = U @ V.T
        diff    = (UV_curr - UV_prev).norm(p="fro").pow(2).item()
        UV_prev = UV_curr
        pbar.set_postfix({"diff": f"{diff:.6f}"})

        if diff < tol:
            pbar.write(f"SLIDE converged at iteration {it + 1}, diff={diff:.2e}")
            break

    # Final orthogonalization per structure block
    r_per_struct = total_r // _NUM_STRUCTURES
    for k in range(_NUM_STRUCTURES):
        col_s = k * r_per_struct
        col_e = col_s + r_per_struct

        U_block = U[:, col_s:col_e]
        V_block = V[:, col_s:col_e]

        try:
            R_b, L_b, Qt_b = torch.linalg.svd(
                U_block @ V_block.T, full_matrices=False
            )
            r_block = U_block.size(1)
            k2      = min(B, p, r_block)
            U[:, col_s:col_e] = R_b[:, :r_block]
            V[:, col_s:col_e] = (L_b[:k2].unsqueeze(1) * Qt_b[:k2]).T[:, :r_block] \
                                 if k2 == r_block else V_block
        except Exception:
            pass

    return U, V  # [B, total_r], [p, total_r]
# ---------------------------------------------------------------------------
# SLIDEFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class SLIDEFusion(FusionModule):
    """
    SLIDE (Structural Learning and Integrative Decomposition) fusion module.

    Wraps the SLIDE iterative matrix factorization algorithm into the
    FusionModule interface so it can be swapped with HCLFusion in EHRModel.

    The structure matrix S is fixed and symmetric: each of the 7 non-empty
    subsets of 3 modalities gets exactly r components, giving total_r = 7 * r
    components.  The final score matrix U of shape [B, 7*r] is passed directly
    to the downstream MLP classifier.

    SLIDE does not have a separate pretraining stage (has_pretrain=False).
    The matrix factorization runs as a forward pass during classifier training.

    Parameters
    ----------
    input_dims : list of input dims per modality, e.g. [512, 512, 512]
    r          : number of components per structure (analogous to HCL latent dim)
    max_iter   : maximum iterations for the SLIDE inner loop
    tol        : convergence tolerance (on ||UV^T diff||_F^2)
    """

    def __init__(
        self,
        input_dims: List[int],
        r: int,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        total_r = _NUM_STRUCTURES * r
        # No pretraining stage for SLIDE
        super().__init__(out_dim=total_r, has_pretrain=False)

        self.r          = r
        self.total_r    = total_r
        self.max_iter   = max_iter
        self.tol        = tol
        self.input_dims = input_dims

        # S is fixed; register as a buffer so it moves with .to(device)
        # Shape: [_NUM_MODALITIES, total_r]
        # Will be properly initialized on first forward (need device info)
        self.register_buffer("S", torch.zeros(_NUM_MODALITIES, total_r))
        self._S_initialized = False

    def _ensure_S(self, device: torch.device):
        """Lazily build S on the correct device on first use."""
        if not self._S_initialized:
            S = _build_S_matrix(self.r, device)
            self.S.copy_(S)
            self._S_initialized = True

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Run SLIDE factorization and return the score matrix U.

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        U : [B, 7*r] tensor
        """
        self._ensure_S(x_list[0].device)

        # Column-center each modality representation (zero-mean across batch)
        x_centered = [xi - xi.mean(dim=0, keepdim=True) for xi in x_list]

        # Unpack tuple; V is not needed in the forward pass
        U, _ = _slide_fit(
            X_list  = x_centered,
            S       = self.S,
            total_r = self.total_r,
            max_iter= self.max_iter,
            tol     = self.tol,
        )
        return U  # [B, 7*r]



    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        **kwargs,
    ) -> Optional[dict]:
        """
        SLIDE has no pretraining stage; always returns None.
        """
        return None
