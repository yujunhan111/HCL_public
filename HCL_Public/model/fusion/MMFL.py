import torch
import numpy as np
from itertools import combinations
from typing import List, Optional, Tuple

from fusion.base import FusionModule


# ---------------------------------------------------------------------------
# Structure table (same as SLIDE/HCL: 7 non-empty subsets of 3 modalities)
# Order: {0,1,2}, {0,1}, {0,2}, {1,2}, {0}, {1}, {2}
# ---------------------------------------------------------------------------
_NUM_MODALITIES = 3
_NUM_STRUCTURES = 7  # 2^3 - 1

_STRUCTURE_MODALITIES = [
    [0, 1, 2],  # global
    [0, 1],     # pairwise 01
    [0, 2],     # pairwise 02
    [1, 2],     # pairwise 12
    [0],        # individual 0
    [1],        # individual 1
    [2],        # individual 2
]


def build_S_matrix(
    p_list: List[int],
    r: int,
) -> np.ndarray:
    """
    Build the binary structure matrix S of shape [p, total_r].

    Each of the 7 structures gets exactly r columns.
    S[feature_row, col] = 1 iff the feature belongs to a modality that
    participates in the corresponding structure.

    Parameters
    ----------
    p_list : list of feature dims per modality, e.g. [512, 512, 512]
    r      : number of components per structure

    Returns
    -------
    S : np.ndarray, shape [sum(p_list), 7 * r], dtype float64
    """
    p_total = sum(p_list)
    total_r = _NUM_STRUCTURES * r
    S = np.zeros((p_total, total_r), dtype=np.float64)

    # Compute row ranges for each modality
    offsets = [0]
    for pi in p_list:
        offsets.append(offsets[-1] + pi)

    for k, mods in enumerate(_STRUCTURE_MODALITIES):
        col_start = k * r
        col_end = col_start + r
        for m in mods:
            row_start = offsets[m]
            row_end = offsets[m + 1]
            S[row_start:row_end, col_start:col_end] = 1.0

    return S


def _mmfl_fit(
    X: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    total_r: int,
    lam: float = 1.0,
    gamma: float = 1.0,
    mu: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    MMFL Algorithm 1: alternating minimization via Augmented Lagrangian.

    Parameters
    ----------
    X       : [n, p] column-centered data matrix
    y       : [n] labels in {-1, +1}
    S       : [p, total_r] binary structure matrix
    total_r : total number of latent components
    lam     : reconstruction weight
    gamma   : beta regularization weight
    mu      : augmented Lagrangian penalty parameter
    max_iter: maximum iterations
    tol     : convergence tolerance on ||U^{k} - U^{k-1}||_F
    verbose : print convergence info

    Returns
    -------
    dict with keys: U, V, beta, b
    """
    from tqdm import tqdm

    n, p = X.shape

    # --- Initialize U via truncated SVD of X ---
    k_svd = min(total_r, n, p)
    try:
        Ur, Sr, Vhr = np.linalg.svd(X, full_matrices=False)
        U = Ur[:, :k_svd]
        if k_svd < total_r:
            extra = np.linalg.qr(
                np.random.randn(n, total_r - k_svd)
            )[0]
            U = np.hstack([U, extra])
    except Exception:
        U = np.linalg.qr(np.random.randn(n, total_r))[0]

    # Initialize other variables
    beta = np.zeros(total_r, dtype=np.float64)
    b = 0.0
    z = np.zeros(n, dtype=np.float64)
    q = np.zeros(n, dtype=np.float64)

    pbar = tqdm(range(max_iter), desc="MMFL iterations", leave=True)

    for it in pbar:
        U_prev = U.copy()

        # Step 1: Update V = X^T U, then enforce structure
        V = X.T @ U                          # [p, total_r]
        V = V * S                             # element-wise mask

        # Step 2: Update beta
        # beta = (mu / (2*gamma + mu)) * U^T (y - b - q/mu - z)
        residual = y - b - q / mu - z         # [n]
        beta = (mu / (2.0 * gamma + mu)) * (U.T @ residual)  # [total_r]

        # Step 3: Update b
        # b = (1/n) * (y - z - U @ beta - q/mu)^T @ 1_n
        b = np.mean(y - z - U @ beta - q / mu)

        # Step 4: Update z
        # s = y - U @ beta - b - q / mu
        s = y - U @ beta - b - q / mu         # [n]
        ys = y * s                             # element-wise
        factor = 1.0 / (1.0 + 2.0 / mu)
        z = np.where(ys > 0, s * factor, s)

        # Step 5: Update U
        # U = (lam * X @ V + (mu/2) * (y - b - z - q/mu) @ beta^T)
        #     @ (lam * V^T @ V + (mu/2) * beta @ beta^T)^{-1}
        # Then enforce orthogonality via SVD
        target = y - b - z - q / mu           # [n]

        # Left matrix: [n, total_r]
        A = lam * (X @ V) + (mu / 2.0) * np.outer(target, beta)

        # Right matrix: [total_r, total_r]
        B = lam * (V.T @ V) + (mu / 2.0) * np.outer(beta, beta)

        # Solve U_raw = A @ B^{-1}
        # More stable: U_raw @ B = A  =>  U_raw = A @ inv(B)
        try:
            B_inv = np.linalg.pinv(B)
            U_raw = A @ B_inv
        except Exception:
            U_raw = A @ np.linalg.pinv(B)

        # Enforce orthogonality: SVD -> U = L @ R^T
        L_svd, _, Rt_svd = np.linalg.svd(U_raw, full_matrices=False)
        U = L_svd @ Rt_svd

        # Step 6: Update dual variable q
        constraint_residual = z - y + U @ beta + b
        q = q + mu * constraint_residual

        # Check convergence
        diff = np.linalg.norm(U - U_prev, ord="fro")
        pbar.set_postfix({"diff": f"{diff:.2e}"})

        if diff < tol:
            if verbose:
                pbar.write(
                    f"MMFL converged at iteration {it + 1}, diff={diff:.2e}"
                )
            break

    return {"U": U, "V": V, "beta": beta, "b": b}

def _mmfl_project(
    X_new: np.ndarray,
    V: np.ndarray,
    S: np.ndarray,
    beta: np.ndarray,
    lam: float
) -> np.ndarray:
    """
    Project out-of-sample data onto the MMFL loading space using Eq (11).
    """
    # Right matrix: (lambda * V^T V + beta * beta^T)
    VtV = V.T @ V
    right_mat = lam * VtV + np.outer(beta, beta)
    right_mat_inv = np.linalg.pinv(right_mat)
    
    # Left matrix: (lambda * X_new @ V)
    left_mat = lam * (X_new @ V)
    
    # U_new = left_mat @ right_mat_inv
    U_new = left_mat @ right_mat_inv
    return U_new

# ---------------------------------------------------------------------------
# MMFLFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class MMFLFusion(FusionModule):
    """
    MMFL (Supervised Multi-Modal Fission Learning) fusion module.

    Unlike SLIDE/HNN, MMFL includes a built-in linear prediction head
    (U @ beta + b) trained jointly with the decomposition, so no
    separate MLP classifier is needed.

    Parameters
    ----------
    input_dims : list of input dims per modality, e.g. [512, 512, 512]
    r          : number of components per structure
    lam        : reconstruction weight in objective
    gamma      : L2 regularization weight for beta
    mu         : augmented Lagrangian penalty parameter
    max_iter   : maximum ALM iterations
    tol        : convergence tolerance
    """

    def __init__(
        self,
        input_dims: List[int],
        r: int,
        lam: float = 1.0,
        gamma: float = 1.0,
        mu: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        total_r = _NUM_STRUCTURES * r
        super().__init__(out_dim=total_r, has_pretrain=False)

        self.r = r
        self.total_r = total_r
        self.input_dims = input_dims
        self.lam = lam
        self.gamma = gamma
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol

        # Stored after fit
        self._V: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None
        self._beta: Optional[np.ndarray] = None
        self._b: Optional[float] = None
        self._col_means: Optional[np.ndarray] = None

    def fit(
        self,
        X_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> np.ndarray:
        """
        Run MMFL on training data.

        Parameters
        ----------
        X_list : [X1, X2, X3] each [N_train, p_d] (CPU tensors)
        labels : [N_train] binary labels 0/1 (CPU tensor)

        Returns
        -------
        U_train : np.ndarray [N_train, total_r]
        """
        # Convert to numpy
        X_np_list = [xi.numpy().astype(np.float64) for xi in X_list]
        p_list = [xi.shape[1] for xi in X_np_list]

        # Build structure matrix S
        self._S = build_S_matrix(p_list, self.r)

        # Concatenate and column-center
        X_all = np.concatenate(X_np_list, axis=1)            # [N, p]
        self._col_means = X_all.mean(axis=0)                  # [p]
        X_centered = X_all - self._col_means

        # If labels are binary {0,1}, convert to {-1,+1};
        # otherwise use raw continuous values (regression setting).
        y_np = labels.numpy().astype(np.float64)
        if np.all((y_np == 0) | (y_np == 1)):
            y_np = 2.0 * y_np - 1.0

        # Run MMFL
        result = _mmfl_fit(
            X=X_centered,
            y=y_np,
            S=self._S,
            total_r=self.total_r,
            lam=self.lam,
            gamma=self.gamma,
            mu=self.mu,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        self._V = result["V"]
        self._beta = result["beta"]
        self._b = result["b"]

        return result["U"]

    def transform(
        self,
        X_list: List[torch.Tensor],
    ) -> np.ndarray:
        """
        Project out-of-sample data onto the training MMFL space.
        """
        assert self._V is not None, "Call fit() on training data first."

        X_np_list = [xi.numpy().astype(np.float64) for xi in X_list]
        X_all = np.concatenate(X_np_list, axis=1)
        X_centered = X_all - self._col_means
        return _mmfl_project(
            X_new=X_centered, 
            V=self._V, 
            S=self._S, 
            beta=self._beta, 
            lam=self.lam
        )
    def predict_proba(self, U: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the built-in linear head.

        Computes sigmoid(U @ beta + b) to output P(y=1).

        Parameters
        ----------
        U : [N, total_r] score matrix (from fit or transform)

        Returns
        -------
        probs : [N] array of probabilities for the positive class
        """
        assert self._beta is not None, "Call fit() first."
        raw = U @ self._beta + self._b                        # [N]
        # Sigmoid to convert to probability
        probs = 1.0 / (1.0 + np.exp(-raw))
        return probs

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Not used in the MMFL training pipeline.
        MMFL requires fit() / transform() / predict_proba() instead.
        """
        raise NotImplementedError(
            "MMFLFusion does not support batch forward(). "
            "Use fit() for train data, transform() for val/test, "
            "and predict_proba() for predictions."
        )

    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        **kwargs,
    ) -> Optional[dict]:
        """MMFL has no pretraining stage; always returns None."""
        return None