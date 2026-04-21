"""
Optimization module for group Lasso regression and related solvers.

This module provides efficient algorithms for solving regularized regression problems:
- group_prox: Proximal operator for group Lasso penalty
- group_lasso_bcd: Block coordinate descent solver
- proximal_gradient_group_lasso: Proximal gradient with acceleration
- compute_aic: Model selection criterion
"""

import torch
import math


@torch.no_grad()
def group_prox(beta, groups, thresh, eps = 1e-12):
    """
    Compute the proximal operator for group Lasso regularization.
    
    For each group g, applies the shrinkage operator:
    beta_g <- max(0, 1 - thresh / ||beta_g||_2) * beta_g
    
    This is the soft-thresholding operator at group level, which encourages
    entire groups to be set to zero for strong sparsity.
    
    Args:
       beta: torch.Tensor, coefficient vector to be processed (shape: r x 1)
       groups: list of lists, each element specifies indices of one group
       thresh: float, threshold parameter (step_size * lambda_m)
       eps: float, small constant to avoid division by zero

    Returns:
       beta: torch.Tensor, coefficient vector after applying the proximal operator
    """
    # Apply group-wise soft thresholding
    for g in groups:
        n = torch.linalg.norm(beta[g], ord=2)
        scale = torch.clamp(1.0 - thresh / (n + eps), min=0.0)
        beta[g] = scale * beta[g]

    return beta

@torch.no_grad()
def group_lasso_bcd(x, y, C_hat, sigma2_hat, lambda_m, groups, device, max_iter=2000, tol=1e-5):
    """
    Solve group Lasso regression using cyclic block coordinate descent (BCD).
    
    Minimizes the objective:
        0.5 * beta^T H beta - g^T beta + lambda_m * sum_s ||beta_s||_2
    where:
        H = (x^T x)/m - sigma2_hat * (C_hat @ C_hat^T)
        g = (x^T y)/m
    
    Algorithm:
    - Cyclic block coordinate descent with Lipschitz-adaptive step sizes
    - For block s: compute gradient, apply proximal operator, update full gradient
    - Lipschitz constant L_s = ||H_ss||_2 (spectral norm of block-diagonal)
    - Step size: 1/L_s (normalized by block Lipschitz constant)
    
    Args:
        x: torch.Tensor, design matrix of shape (m, r)
        y: torch.Tensor, target vector of shape (m,) or (m, 1)
        C_hat: torch.Tensor, transformation matrix of shape (r, d)
        sigma2_hat: float, estimated noise variance
        lambda_m: float, group Lasso regularization parameter
        groups: list of lists, indices for each group in beta
        device: str, computing device ("cuda" or "cpu")
        max_iter: int, maximum number of iterations
        tol: float, convergence tolerance (relative change threshold)

    Returns:
        beta: torch.Tensor, optimized coefficient vector of shape (r, 1)
    """
    m, r = x.shape
    H = (x.T@x)/m -sigma2_hat* (C_hat@C_hat.T)
    b = (x.T@y)/m
    beta = torch.zeros((r, 1), device=device)

    # Full gradient grad = H beta - b
    grad = H @ beta - b

    # Precompute Lipschitz constants for each group (spectral norm of H_ss)
    L_groups = []
    for idx in groups:
        H_ss = H[idx][:, idx]
        eigs = torch.linalg.eigvalsh(H_ss)
        Ls = eigs.abs().max().clamp_min(1e-8)
        L_groups.append(Ls)

    # Cyclic block coordinate descent iterations
    for epoch in range(max_iter):
        max_rel = 0.0

        for s, idx in enumerate(groups):
            old = beta[idx].clone()
            Ls = L_groups[s]
            grad_s = grad[idx]

            # Majorized / proximal block step
            temp = old - grad_s / Ls
            # Apply group Lasso proximal operator
            new = group_prox(temp, [torch.arange(temp.numel(), device=temp.device)], lambda_m / Ls)

            # Update beta and gradient if there's any change
            delta = new - old
            if torch.linalg.norm(delta) > 0:
                beta[idx] = new
                grad = grad + H[:, idx] @ delta

                # Compute relative change for convergence
                rel = (torch.linalg.norm(delta) / torch.linalg.norm(old).clamp_min(1.0)).item()
                if rel > max_rel:
                    max_rel = rel

        # Check convergence: stop if maximum relative change is below tolerance
        if max_rel < tol:
            break

    return beta

@torch.no_grad()
def proximal_gradient_group_lasso(x, y, C_hat, sigma2_hat, lambda_m, groups,  device, max_iter=2000, tol=1e-5, lr=1e-3,):
    """
    Solve the group Lasso regression using accelerated proximal gradient method.
    
    Minimizes:
        (1/2m)||y - Phi beta||^2 + (1/2m) sigma2_hat * beta^T (C_hat C_hat^T) beta
        + lambda_m * sum_s ||beta_s||_2
    
    Algorithm:
    - Extrapolation-based acceleration (similar to FISTA)
    - Updates: 
        1. Gradient step on smooth part
        2. Proximal step for group Lasso penalty
        3. Momentum update with coefficients t = (1 + sqrt(1+4t_old^2))/2
    
    Args:
        x: torch.Tensor, design matrix of shape (m, r)
        y: torch.Tensor, target vector of shape (m,) or (m, 1)
        C_hat: torch.Tensor, transformation matrix of shape (r, d)
        sigma2_hat: float, estimated noise variance
        lambda_m: float, group Lasso regularization parameter
        groups: list of lists, indices defining groups in beta
        device: str, computing device ("cuda" or "cpu")
        max_iter: int, maximum number of iterations
        tol: float, convergence tolerance
        lr: float, learning rate (step size). If None, computed from Lipschitz constant.

    Returns:
        beta: torch.Tensor, optimized coefficient vector of shape (r, 1)
    """

    m, r = x.shape
    H = (x.T@x)/m -sigma2_hat* (C_hat@C_hat.T)
    b = (x.T@y)/m

    # Compute learning rate from Lipschitz constant if not provided
    if lr is None:
        # Lipschitz constant = spectral norm of H
        L = torch.linalg.eigvalsh(H).abs().max().clamp_min(1e-8)
        lr = (1.0/L).item()

    # Initialize primal and dual variables
    beta = torch.zeros((r, 1), device=device)
    z = beta.clone()  # Extrapolation variable
    t = 1.0           # Acceleration parameter

    for k in range(max_iter):
        beta_old = beta.clone()
        grad = H @ z - b

        # Gradient descent step for smooth part
        beta = z - lr * grad
        # Proximal step for group Lasso penalty
        beta = group_prox(beta, groups, thresh=lr * lambda_m)

        # Acceleration: Nesterov-style momentum update
        t_new = 0.5*(1+(1+4*t*t)**0.5)
        z = beta+((t-1)/t_new)*(beta-beta_old)
        t = t_new

        # Stopping criterion: check convergence of beta
        if torch.linalg.norm(beta - beta_old) < tol:
            break

    return beta

def compute_aic(y, x, beta, groups, eps=1e-12):
    """
    Compute AIC (Akaike Information Criterion) for model selection.
    
    AIC = -2 * log(RSS) + k * log(n)
    where:
        RSS: residual sum of squares
        k: effective degrees of freedom (number of non-zero groups)
        n: number of samples
    
    Args:
        y: torch.Tensor, target vector
        x: torch.Tensor, design matrix
        beta: torch.Tensor, coefficient vector
        groups: list of lists, group indices
        eps: float, small constant for numerical stability
    
    Returns:
        aic: float, AIC value
    """
    n = y.shape[0]
    res = y - x @ beta
    rss = torch.sum(res ** 2)

    df = 0
    for g in groups:
        if torch.norm(beta[g], p=2) > 1e-8:
            df += len(g)

    # Compute AIC
    aic = -2 * torch.log(rss + eps) + math.log(n) * df
    return aic.item()

def make_folds(m, n_folds=3, seed=0):
    """Split data into folds for cross-validation.

    Args:
        m: int, total number of samples.
        n_folds: int, number of folds.
        seed: int, random seed.

    Returns:
        folds: list, each element is an index tensor representing one fold.
    """
    torch.manual_seed(seed)
    indices = torch.randperm(m)
    folds = torch.chunk(indices, n_folds)

    return folds