import torch
import numpy as np
from typing import List, Optional

from fusion.base import FusionModule


# ---------------------------------------------------------------------------
# Core sJIVE alternating optimization algorithm
# ---------------------------------------------------------------------------
def _truncated_svd_torch(M: torch.Tensor, k: int):
    """
    Truncated SVD on GPU via torch.
    Falls back to full SVD if k >= min(M.shape).
    """
    m, n = M.shape
    if k >= min(m, n):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k, :]
    U, S, V = torch.svd_lowrank(M, q=k, niter=4)
    return U, S, V.T


def _sjive_fit(
    X_list: List[torch.Tensor],
    y: torch.Tensor,
    joint_rank: int,
    indiv_ranks: List[int],
    eta: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    sJIVE alternating estimation — GPU + truncated SVD version.

    Parameters
    ----------
    X_list      : list of k tensors, each [p_i, n] on device, column-centered
    y           : [n] tensor on device, centered outcome
    joint_rank  : rank of joint structure
    indiv_ranks : list of k ints
    eta         : weight balancing reconstruction vs prediction

    Returns
    -------
    dict with U_joint, U_indiv, V_joint, V_indiv, theta_1, theta_2 (all torch on device)
    """
    from tqdm import tqdm

    k = len(X_list)
    device = X_list[0].device
    dtype = X_list[0].dtype
    n = X_list[0].size(1)
    p_list = [Xi.size(0) for Xi in X_list]
    p_total = sum(p_list)

    w_data = (eta ** 0.5)
    w_pred = ((1.0 - eta) ** 0.5)

    # Initialize
    S_J = torch.zeros(joint_rank, n, device=device, dtype=dtype)
    S_i_list = [torch.zeros(indiv_ranks[i], n, device=device, dtype=dtype) for i in range(k)]

    U_list = [torch.zeros(p_list[i], joint_rank, device=device, dtype=dtype) for i in range(k)]
    W_list = [torch.zeros(p_list[i], indiv_ranks[i], device=device, dtype=dtype) for i in range(k)]
    theta_1 = torch.zeros(joint_rank, device=device, dtype=dtype)
    theta_2 = [torch.zeros(indiv_ranks[i], device=device, dtype=dtype) for i in range(k)]

    pbar = tqdm(range(max_iter), desc="sJIVE iterations", leave=True)

    for it in pbar:
        S_J_prev = S_J.clone()

        # === Step 1: Update joint scores S_J ===
        residual_data = []
        for i in range(k):
            res_i = X_list[i] - W_list[i] @ S_i_list[i]
            residual_data.append(w_data * res_i)

        y_indiv_pred = torch.zeros(n, device=device, dtype=dtype)
        for i in range(k):
            if indiv_ranks[i] > 0:
                y_indiv_pred = y_indiv_pred + theta_2[i] @ S_i_list[i]
        residual_y = (w_pred * (y - y_indiv_pred)).unsqueeze(0)  # [1, n]

        augmented_joint = torch.cat(residual_data + [residual_y], dim=0)  # [p_total + 1, n]

        if joint_rank > 0 and min(augmented_joint.shape) > 0:
            U_aug, S_aug, Vt_aug = _truncated_svd_torch(augmented_joint, joint_rank)
            r_eff = S_aug.size(0)
            S_J = Vt_aug  # [r_eff, n]
            if r_eff < joint_rank:
                S_J = torch.cat([
                    S_J,
                    torch.zeros(joint_rank - r_eff, n, device=device, dtype=dtype),
                ], dim=0)

            # Extract loadings
            U_aug_scaled = U_aug * S_aug.unsqueeze(0)  # [p_total+1, r_eff]
            offset = 0
            for i in range(k):
                U_list[i] = U_aug_scaled[offset:offset + p_list[i], :r_eff] / (w_data + 1e-12)
                if r_eff < joint_rank:
                    U_list[i] = torch.cat([
                        U_list[i],
                        torch.zeros(p_list[i], joint_rank - r_eff, device=device, dtype=dtype),
                    ], dim=1)
                offset += p_list[i]
            theta_1_vec = U_aug_scaled[offset:offset + 1, :r_eff].squeeze(0) / (w_pred + 1e-12)
            if r_eff < joint_rank:
                theta_1 = torch.cat([theta_1_vec, torch.zeros(joint_rank - r_eff, device=device, dtype=dtype)])
            else:
                theta_1 = theta_1_vec
        else:
            S_J = torch.zeros(joint_rank, n, device=device, dtype=dtype)

        # === Step 2: Update individual scores S_i ===
        for i in range(k):
            ri = indiv_ranks[i]
            if ri <= 0:
                continue

            res_data_i = w_data * (X_list[i] - U_list[i] @ S_J)

            y_other_pred = theta_1 @ S_J
            for j in range(k):
                if j != i and indiv_ranks[j] > 0:
                    y_other_pred = y_other_pred + theta_2[j] @ S_i_list[j]
            res_y_i = (w_pred * (y - y_other_pred)).unsqueeze(0)

            augmented_indiv = torch.cat([res_data_i, res_y_i], dim=0)

            U_ai, S_ai, Vt_ai = _truncated_svd_torch(augmented_indiv, ri)
            r_eff = S_ai.size(0)
            S_i_raw = Vt_ai  # [r_eff, n]

            # Enforce orthogonality with S_J
            if joint_rank > 0 and S_J.norm() > 1e-10:
                SJSJt = S_J @ S_J.T
                SJSJt_inv = torch.linalg.pinv(SJSJt)
                proj_coeff = S_i_raw @ S_J.T @ SJSJt_inv
                S_i_raw = S_i_raw - proj_coeff @ S_J

            # Re-orthonormalize
            if S_i_raw.size(0) > 0 and S_i_raw.norm() > 1e-10:
                Q, _ = torch.linalg.qr(S_i_raw.T)
                S_i_raw = Q[:, :r_eff].T

            if r_eff < ri:
                S_i_raw = torch.cat([
                    S_i_raw,
                    torch.zeros(ri - r_eff, n, device=device, dtype=dtype),
                ], dim=0)
            S_i_list[i] = S_i_raw

            # Extract loadings and theta
            U_ai_scaled = U_ai * S_ai.unsqueeze(0)
            W_list[i] = U_ai_scaled[:p_list[i], :r_eff] / (w_data + 1e-12)
            theta_2_vec = U_ai_scaled[p_list[i]:p_list[i] + 1, :r_eff].squeeze(0) / (w_pred + 1e-12)
            if r_eff < ri:
                W_list[i] = torch.cat([
                    W_list[i],
                    torch.zeros(p_list[i], ri - r_eff, device=device, dtype=dtype),
                ], dim=1)
                theta_2[i] = torch.cat([theta_2_vec, torch.zeros(ri - r_eff, device=device, dtype=dtype)])
            else:
                theta_2[i] = theta_2_vec

        # === Convergence ===
        diff = (S_J - S_J_prev).norm(p="fro").item()
        pbar.set_postfix({"diff": f"{diff:.2e}"})

        if diff < tol:
            if verbose:
                pbar.write(f"sJIVE converged at iteration {it + 1}, diff={diff:.2e}")
            break

    # --- Final loadings ---
    X_stacked = torch.cat(X_list, dim=0)

    if joint_rank > 0 and S_J.norm() > 1e-10:
        SJSJt_inv = torch.linalg.pinv(S_J @ S_J.T)
        V_joint = X_stacked @ S_J.T @ SJSJt_inv
    else:
        V_joint = torch.zeros(p_total, joint_rank, device=device, dtype=dtype)

    V_indiv_list = []
    for i in range(k):
        ri = indiv_ranks[i]
        if ri > 0 and S_i_list[i].norm() > 1e-10:
            SiSit_inv = torch.linalg.pinv(S_i_list[i] @ S_i_list[i].T)
            V_indiv_i = X_list[i] @ S_i_list[i].T @ SiSit_inv
        else:
            V_indiv_i = torch.zeros(p_list[i], ri, device=device, dtype=dtype)
        V_indiv_list.append(V_indiv_i)

    U_joint = S_J.T   # [n, joint_rank]
    U_indiv = [S_i_list[i].T for i in range(k)]

    return {
        "U_joint": U_joint,
        "U_indiv": U_indiv,
        "V_joint": V_joint,
        "V_indiv": V_indiv_list,
        "theta_1": theta_1,
        "theta_2": theta_2,
    }

def _sjive_project(
    X_list: List[torch.Tensor],
    V_joint: torch.Tensor,
    V_indiv_list: List[torch.Tensor],
    S_J_basis: torch.Tensor,
    joint_rank: int,
    indiv_ranks: List[int],
) -> torch.Tensor:
    """
    Project out-of-sample data — torch version.
    """
    n_new = X_list[0].size(1)
    device = X_list[0].device
    dtype = X_list[0].dtype

    X_stacked = torch.cat(X_list, dim=0)

    if joint_rank > 0 and V_joint.norm() > 1e-10:
        VjVj = V_joint.T @ V_joint
        VjVj_inv = torch.linalg.pinv(VjVj)
        U_new_joint = X_stacked.T @ V_joint @ VjVj_inv
    else:
        U_new_joint = torch.zeros(n_new, joint_rank, device=device, dtype=dtype)

    U_new_indiv = []
    for i in range(len(X_list)):
        ri = indiv_ranks[i]
        Vi = V_indiv_list[i]

        if ri > 0 and Vi.norm() > 1e-10:
            ViVi = Vi.T @ Vi
            ViVi_inv = torch.linalg.pinv(ViVi)
            U_new_i = X_list[i].T @ Vi @ ViVi_inv

            if joint_rank > 0 and U_new_joint.norm() > 1e-10:
                JtJ = U_new_joint.T @ U_new_joint
                JtJ_inv = torch.linalg.pinv(JtJ)
                proj_coeff = JtJ_inv @ U_new_joint.T @ U_new_i
                U_new_i = U_new_i - U_new_joint @ proj_coeff
        else:
            U_new_i = torch.zeros(n_new, ri, device=device, dtype=dtype)

        U_new_indiv.append(U_new_i)

    return torch.cat([U_new_joint] + U_new_indiv, dim=1)
# ---------------------------------------------------------------------------
# sJIVEFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class sJIVEFusion(FusionModule):

    def __init__(
        self,
        input_dims: List[int],
        r: int,
        eta: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        total_r = 4 * r
        super().__init__(out_dim=total_r, has_pretrain=False)

        self.r = r
        self.total_r = total_r
        self.input_dims = input_dims
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

        # Stored after fit (all torch tensors on CPU)
        self._V_joint: Optional[torch.Tensor] = None
        self._V_indiv: Optional[List[torch.Tensor]] = None
        self._col_means: Optional[List[torch.Tensor]] = None
        self._y_mean: Optional[float] = None
        self._S_J: Optional[torch.Tensor] = None

    def fit(
        self,
        X_list: List[torch.Tensor],
        labels: torch.Tensor,
        device: torch.device = None,
    ) -> np.ndarray:
        """
        Run sJIVE on training data (GPU accelerated).

        Parameters
        ----------
        X_list : [X1, X2, X3] each [N_train, p_d] (CPU tensors)
        labels : [N_train] binary labels 0/1 (CPU tensor)
        device : torch.device for GPU acceleration

        Returns
        -------
        U_train : np.ndarray [N_train, total_r]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transpose to [p_d, n] and move to GPU
        X_gpu_list = [xi.T.to(device).double() for xi in X_list]

        # Column-center
        self._col_means = [Xi.mean(dim=1, keepdim=True).cpu() for Xi in X_gpu_list]
        X_centered = [Xi - Xi.mean(dim=1, keepdim=True) for Xi in X_gpu_list]

        # If labels are binary {0,1}, convert to {-1,+1};
        # otherwise use raw continuous values (regression setting).
        y_gpu = labels.double().to(device)
        if torch.all((y_gpu == 0) | (y_gpu == 1)):
            y_gpu = 2.0 * y_gpu - 1.0
        self._y_mean = y_gpu.mean().item()
        y_centered = y_gpu - self._y_mean

        # Run sJIVE on GPU
        indiv_ranks = [self.r] * len(X_list)
        result = _sjive_fit(
            X_list=X_centered,
            y=y_centered,
            joint_rank=self.r,
            indiv_ranks=indiv_ranks,
            eta=self.eta,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        # Store on CPU
        self._V_joint = result["V_joint"].cpu()
        self._V_indiv = [v.cpu() for v in result["V_indiv"]]
        self._S_J = result["U_joint"].T.cpu()  # [joint_rank, n]

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
        Project out-of-sample data (GPU accelerated).

        Parameters
        ----------
        X_list : [X1, X2, X3] each [N, p_d] (CPU tensors)
        device : torch.device

        Returns
        -------
        U_new : np.ndarray [N, total_r]
        """
        assert self._V_joint is not None, "Call fit() on training data first."

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_gpu_list = [xi.T.to(device).double() for xi in X_list]

        X_centered = [
            Xi - mu.to(device)
            for Xi, mu in zip(X_gpu_list, self._col_means)
        ]

        indiv_ranks = [self.r] * len(X_list)
        U_new = _sjive_project(
            X_list=X_centered,
            V_joint=self._V_joint.to(device),
            V_indiv_list=[v.to(device) for v in self._V_indiv],
            S_J_basis=self._S_J.to(device),
            joint_rank=self.r,
            indiv_ranks=indiv_ranks,
        )
        return U_new.cpu().float().numpy()

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "sJIVEFusion does not support batch forward(). "
            "Use fit() for train data and transform() for val/test."
        )

    def compute_pretrain_loss(self, x_list, **kwargs):
        return None