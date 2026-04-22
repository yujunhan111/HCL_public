import torch
import numpy as np
from typing import List, Optional

from fusion.base import FusionModule


# ---------------------------------------------------------------------------
# Core JIVE alternating SVD algorithm
# ---------------------------------------------------------------------------
def _truncated_svd_torch(M: torch.Tensor, k: int):
    """
    Truncated SVD on GPU via torch.
    Falls back to full SVD if k >= min(M.shape).

    Parameters
    ----------
    M : torch.Tensor on GPU/CPU
    k : number of singular components to keep

    Returns
    -------
    U : [m, k], S : [k], Vt : [k, n]
    """
    m, n = M.shape
    if k >= min(m, n):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k, :]
    # torch.svd_lowrank is randomized truncated SVD — much faster for large matrices
    U, S, V = torch.svd_lowrank(M, q=k, niter=4)
    # U: [m, k], S: [k], V: [n, k]
    return U, S, V.T  # Vt: [k, n]

def _jive_fit(
    X_list: List[torch.Tensor],
    joint_rank: int,
    indiv_ranks: List[int],
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    JIVE alternating estimation algorithm — GPU + truncated SVD version.

    Now operates on torch.Tensor (GPU) instead of numpy arrays.
    Input X_list: each [p_i, n] torch.Tensor on device, already column-centered.

    Returns dict with U_joint, U_indiv, V_joint, V_indiv (all torch.Tensor on same device).
    """
    from tqdm import tqdm

    k = len(X_list)
    device = X_list[0].device
    dtype = X_list[0].dtype
    n = X_list[0].size(1)
    p_list = [Xi.size(0) for Xi in X_list]

    # Initialize: J_i = 0
    J_list = [torch.zeros_like(Xi) for Xi in X_list]
    A_list = [torch.zeros_like(Xi) for Xi in X_list]

    # Row offsets for stacked matrix
    offsets = [0]
    for pi in p_list:
        offsets.append(offsets[-1] + pi)

    pbar = tqdm(range(max_iter), desc="JIVE iterations", leave=True)

    for it in pbar:
        J_prev = [Ji.clone() for Ji in J_list]

        # --- Step 1: Fix J, estimate A_i ---
        for i in range(k):
            residual_i = X_list[i] - J_list[i]  # [p_i, n]
            ri = indiv_ranks[i]
            if ri > 0 and min(residual_i.shape) > 0:
                U_i, S_i, Vt_i = _truncated_svd_torch(residual_i, ri)
                A_list[i] = (U_i * S_i.unsqueeze(0)) @ Vt_i
            else:
                A_list[i] = torch.zeros_like(X_list[i])

        # --- Step 2: Fix A, estimate J ---
        stacked = torch.cat([X_list[i] - A_list[i] for i in range(k)], dim=0)  # [sum(p_i), n]

        if joint_rank > 0 and min(stacked.shape) > 0:
            U_s, S_s, Vt_s = _truncated_svd_torch(stacked, joint_rank)
            J_stacked = (U_s * S_s.unsqueeze(0)) @ Vt_s

            for i in range(k):
                J_list[i] = J_stacked[offsets[i]:offsets[i + 1], :]
        else:
            for i in range(k):
                J_list[i] = torch.zeros_like(X_list[i])

        # --- Step 3: Enforce orthogonality ---
        for i in range(k):
            if J_list[i].size(0) > 0 and J_list[i].norm() > 1e-10:
                _, _, Vt_Ji = _truncated_svd_torch(J_list[i], joint_rank)
                V_Ji = Vt_Ji.T  # [n, r_Ji]
                proj = A_list[i] @ V_Ji @ V_Ji.T
                A_list[i] = A_list[i] - proj

        # --- Convergence check ---
        diff = sum(
            (J_list[i] - J_prev[i]).norm(p="fro").item()
            for i in range(k)
        )
        pbar.set_postfix({"diff": f"{diff:.2e}"})

        if diff < tol:
            if verbose:
                pbar.write(f"JIVE converged at iteration {it + 1}, diff={diff:.2e}")
            break

    # --- Extract score matrices ---
    J_full = torch.cat(J_list, dim=0)  # [sum(p_i), n]
    if joint_rank > 0 and J_full.norm() > 1e-10:
        U_j, S_j, Vt_j = _truncated_svd_torch(J_full, joint_rank)
        U_joint = Vt_j.T * S_j.unsqueeze(0)  # [n, joint_rank]
        V_joint = U_j  # [sum(p_i), joint_rank]
    else:
        U_joint = torch.zeros(n, joint_rank, device=device, dtype=dtype)
        V_joint = torch.zeros(sum(p_list), joint_rank, device=device, dtype=dtype)

    # Individual scores
    U_indiv_list = []
    V_indiv_list = []
    for i in range(k):
        ri = indiv_ranks[i]
        if ri > 0 and A_list[i].norm() > 1e-10:
            U_ai, S_ai, Vt_ai = _truncated_svd_torch(A_list[i], ri)
            U_indiv_i = Vt_ai.T * S_ai.unsqueeze(0)  # [n, ri]
            V_indiv_i = U_ai  # [p_i, ri]
        else:
            U_indiv_i = torch.zeros(n, ri, device=device, dtype=dtype)
            V_indiv_i = torch.zeros(p_list[i], ri, device=device, dtype=dtype)
        U_indiv_list.append(U_indiv_i)
        V_indiv_list.append(V_indiv_i)

    return {
        "J_list": J_list,
        "A_list": A_list,
        "U_joint": U_joint,
        "U_indiv": U_indiv_list,
        "V_joint": V_joint,
        "V_indiv": V_indiv_list,
    }



def _jive_project(
    X_list: List[torch.Tensor],
    V_joint: torch.Tensor,
    V_indiv_list: List[torch.Tensor],
    joint_rank: int,
    indiv_ranks: List[int],
) -> torch.Tensor:
    """
    Project out-of-sample data onto the JIVE loading space — torch version.

    Parameters
    ----------
    X_list       : list of k tensors, each [p_i, n_new] on device, centered with training means
    V_joint      : [sum(p_i), joint_rank] on device
    V_indiv_list : list of k tensors [p_i, indiv_ranks[i]] on device

    Returns
    -------
    U_new : torch.Tensor [n_new, joint_rank + sum(indiv_ranks)]
    """
    n_new = X_list[0].size(1)

    # Joint projection
    X_stacked = torch.cat(X_list, dim=0)  # [sum(p_i), n_new]
    U_new_joint = X_stacked.T @ V_joint  # [n_new, joint_rank]

    # Individual projections
    U_new_indiv = []
    for i in range(len(X_list)):
        U_new_i = X_list[i].T @ V_indiv_list[i]  # [n_new, indiv_ranks[i]]
        U_new_indiv.append(U_new_i)

    return torch.cat([U_new_joint] + U_new_indiv, dim=1)  # [n_new, total_r]

# ---------------------------------------------------------------------------
# JIVEFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class JIVEFusion(FusionModule):

    def __init__(
        self,
        input_dims: List[int],
        r: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        total_r = 4 * r
        super().__init__(out_dim=total_r, has_pretrain=False)

        self.r = r
        self.total_r = total_r
        self.input_dims = input_dims
        self.max_iter = max_iter
        self.tol = tol

        # Stored after fit (all torch tensors on CPU)
        self._V_joint: Optional[torch.Tensor] = None
        self._V_indiv: Optional[List[torch.Tensor]] = None
        self._col_means: Optional[List[torch.Tensor]] = None

    def fit(
        self,
        X_list: List[torch.Tensor],
        device: torch.device = None,
    ) -> np.ndarray:
        """
        Run JIVE on training data.

        Parameters
        ----------
        X_list : [X1, X2, X3] each [N_train, p_d] (CPU tensors)
        device : torch.device for GPU acceleration (optional, defaults to cuda if available)

        Returns
        -------
        U_train : np.ndarray [N_train, total_r]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transpose to [p_d, n] and move to GPU
        X_gpu_list = [xi.T.to(device).double() for xi in X_list]

        # Column-center (subtract row means)
        self._col_means = [Xi.mean(dim=1, keepdim=True).cpu() for Xi in X_gpu_list]
        X_centered = [Xi - Xi.mean(dim=1, keepdim=True) for Xi in X_gpu_list]

        # Run JIVE on GPU
        indiv_ranks = [self.r] * len(X_list)
        result = _jive_fit(
            X_list=X_centered,
            joint_rank=self.r,
            indiv_ranks=indiv_ranks,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        # Store loadings on CPU
        self._V_joint = result["V_joint"].cpu()
        self._V_indiv = [v.cpu() for v in result["V_indiv"]]

        # Concatenate scores and return as numpy
        U_train = torch.cat(
            [result["U_joint"]] + result["U_indiv"], dim=1
        ).cpu().float().numpy()
        return U_train

    def transform(
        self,
        X_list: List[torch.Tensor],
        device: torch.device = None,
    ) -> np.ndarray:
        """
        Project out-of-sample data onto the training JIVE space.

        Parameters
        ----------
        X_list : [X1, X2, X3] each [N, p_d] (CPU tensors)
        device : torch.device for GPU acceleration

        Returns
        -------
        U_new : np.ndarray [N, total_r]
        """
        assert self._V_joint is not None, "Call fit() on training data first."

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transpose to [p_d, n] and move to GPU
        X_gpu_list = [xi.T.to(device).double() for xi in X_list]

        # Apply training centering
        X_centered = [
            Xi - mu.to(device)
            for Xi, mu in zip(X_gpu_list, self._col_means)
        ]

        indiv_ranks = [self.r] * len(X_list)
        U_new = _jive_project(
            X_list=X_centered,
            V_joint=self._V_joint.to(device),
            V_indiv_list=[v.to(device) for v in self._V_indiv],
            joint_rank=self.r,
            indiv_ranks=indiv_ranks,
        )
        return U_new.cpu().float().numpy()

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "JIVEFusion does not support batch forward(). "
            "Use fit() for train data and transform() for val/test."
        )

    def compute_pretrain_loss(self, x_list, **kwargs):
        return None
