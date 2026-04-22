"""
Downstream linear regression task with variable representation learning sample size (n).

This module evaluates the performance of learned representations on a downstream
linear regression task by varying the sample size of representation learning data (n)
while keeping the downstream task data size fixed (m=10000).

Methods compared:
- OLS (Ordinary Least Squares) estimator
- Group Lasso estimator with cross-validation
- Multiple runs with averaging for stability
"""

import argparse
from Data_generation import setup_seed, Data_generate, Label_generate
import numpy as np
from scipy.linalg import orthogonal_procrustes
import torch
from HCL import HCL_SVD, HCL_grad
from Solve import group_lasso_bcd, proximal_gradient_group_lasso

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", nargs='+', type=int, help="Dimension of raw data for three modality")
    parser.add_argument("--r", default=5, type=int, help="Dimension of latent vector")
    parser.add_argument("--c", default=0.1, type=float, help="noise level")
    parser.add_argument("--dvc", default='cuda', type=str, help="device used")
    return parser


def make_folds(m, n_folds=3, seed=0):
    """
    Split data into folds for cross-validation.
    """
    torch.manual_seed(seed)
    indices = torch.randperm(m)
    folds = torch.chunk(indices, n_folds)

    return folds


def GL_estimator(x, y, C_hat, sigma2_hat, groups, device):
    """
    Perform K-fold cross-validation to select optimal group Lasso parameter.
    
    Evaluates multiple regularization parameters using 3-fold cross-validation,
    then fits the final model using the parameter with minimum CV error.
    
    Args:
        x: torch.Tensor, design matrix of shape (m, r).
        y: torch.Tensor, target vector of shape (m, 1).
        C_hat: torch.Tensor, transformation matrix of shape (r, d).
        sigma2_hat: float, estimated noise variance.
        groups: list, each element is an index list defining groups in beta.
        device: str, computing device ("cuda" or "cpu").

    Returns:
        best_beta: torch.Tensor, coefficient estimate with optimal regularization of shape (r, 1).
    """
    m = x.shape[0]
    folds = make_folds(m, n_folds=3)
    cv_errors = []
    # Grid of regularization parameters: 10^{-6} to 10^0
    lambda_list = [10**i for i in np.arange(-6,1, dtype=np.float32)]
    for alpha in lambda_list:
        fold_scores = []

        for k in range(3):
            val_idx = folds[k]
            train_idx = torch.cat([folds[i] for i in range(3) if i != k])
            x_train, y_train, x_val, y_val = x[train_idx], y[train_idx], x[val_idx], y[val_idx]
            beta_hat = group_lasso_bcd(x, y, C_hat, sigma2_hat, alpha, groups, device=device)

            # Evaluate on validation fold: compute MSE
            residual = x_val@beta_hat-y_val
            score = (residual.T@residual)/y_val.shape[0]
            fold_scores.append(score.reshape(-1).item())

        cv_errors.append(np.mean(fold_scores))

    # Select lambda with minimum CV error
    best_index = np.argmin(cv_errors)
    best_lambda = lambda_list[best_index]
    best_beta = group_lasso_bcd(x, y, C_hat, sigma2_hat, best_lambda, groups, device=device)

    return best_beta.reshape(-1, 1)


def Downstream_metric(beta, beta_hat, C, C_hat, r, x, y):
    """
    Compute downstream task error metrics comparing true and estimated coefficients.
    
    Metrics include:
    - Group-wise L2 distances: ||beta_hat_g - beta_g||_2 for 7 groups
    - Overall L2 distance: ||beta_hat - beta||_2
    - Excess prediction risk: E[(y - x*C_hat^T*beta_hat)^2] - E[(y - x*C^T*beta)^2]
    
    Args:
        beta: torch.Tensor, true coefficient vector
        beta_hat: torch.Tensor, estimated coefficient vector
        C: torch.Tensor, true transformation matrix
        C_hat: torch.Tensor, estimated transformation matrix
        r: int, rank of latent components
        x: torch.Tensor, test data
        y: torch.Tensor, test labels
    
    Returns:
        metrics: torch.Tensor, vector of 9 error metrics
    """
    # Compute excess prediction risk
    Risk = torch.mean((y - x@C_hat.T@beta_hat)**2) - torch.mean((y - x@C.T@beta)**2)
    dis = torch.linalg.norm(beta_hat-beta, 2)
    dis_g = [torch.linalg.norm(beta_hat[i*r:(i+1)*r]-beta[i*r:(i+1)*r], 2) for i in range(7)]
    res = dis_g+[dis, Risk]

    return torch.tensor(res)


def run_one_rep(ii, n, c, d1, d2, d3, r, dvc):
    """
    Run one repetition of downstream regression experiment with variable representation data.
    
    Procedure:
    1. Generate representation learning data with sample size n
    2. Estimate noise variance and clean covariance
    3. Learn weight matrix using HCL-SVD and gradient refinement (20 runs with averaging)
    4. On fixed downstream task (m=10000): fit OLS and Group Lasso (20 runs with averaging)
    5. Evaluate using L2 distances and excess risk
    
    Args:
        ii: int, repetition index
        n: int, sample size for representation learning
        c: float, observation noise level
        d1, d2, d3: int, modality dimensions
        r: int, latent rank
        dvc: str, device
    
    Returns:
        (err1, err2): tuple of error vectors for OLS and Group Lasso methods
    """
    # Set random seed
    setup_seed(20+ii)

    d = d1+d2+d3
    m = 10000  # Fixed downstream task sample size
    # Representation learning data (variable size)
    x1, W = Data_generate(d1, d2, d3, r=r, n=n, c=c, device=dvc)
    # Downstream task data (split into train and test)
    x2, y, beta = Label_generate(W, m=2*m, d1=d1, d2=d2, d3=d3, r=r, c=c, cy=0.1, device=dvc)
    y, beta = y.reshape(-1, 1), beta.reshape(-1, 1)
    S_n = torch.cov(x1.T)  # sample covariance
    eigvals = torch.linalg.eigvalsh(S_n)  # ascending order
    sigma2_hat = eigvals[:(d-7*r)].mean()
    S_W = S_n - sigma2_hat*torch.eye(d, device=dvc)  # Denoised covariance

    # Create mask matrix for hierarchical structure
    mask = np.ones((7*r, d), dtype=np.float32)
    mask[np.r_[3*r:4*r, 5*r:7*r], :d1] = 0
    mask[np.r_[2*r:3*r, 4*r:5*r, 6*r:7*r], d1:(d1+d2)] = 0
    mask[np.r_[r:2*r, 4*r:6*r], -d3:] = 0
    mask = torch.tensor(mask, device=dvc)

    # Learn weight matrix: average over 20 runs for stability
    W_ls = []
    for _ in range(20):
        W_svd = HCL_SVD(S_W, d1, d2, d3, r, device=dvc)
        V_grad = HCL_grad(S_W, W_initial=W_svd, mask=mask)
        R = orthogonal_procrustes(V_grad.cpu().numpy().T, W.cpu().numpy())[0]
        R = torch.tensor(R, device=dvc)
        W_grad = V_grad.T@R
        W_ls.append(W_grad)
    W_grad = sum(W_ls) / len(W_ls)  # Average weight matrix

    # Downstream task: prepare input features
    C_true = torch.linalg.inv(W.T@W)@W.T
    C_hat = torch.linalg.inv(W_grad.T@W_grad)@W_grad.T
    X2 = x2[:m] @ C_hat.T  # Transform to latent space

    # Method 1: OLS estimator
    A = (X2.T@X2)/m - sigma2_hat*(C_hat@C_hat.T)
    beta_ls = torch.linalg.solve(A, (X2.T@y[:m])/m)

    # Method 2: Group Lasso estimator (average over 20 runs)
    groups = [list(range(i*r, (i+1)*r)) for i in range(7)]
    beta_la_ls = []
    for _ in range(20):
        beta_la = GL_estimator(X2, y[:m], C_hat, sigma2_hat, groups, device=dvc)
        beta_la_ls.append(beta_la)
    beta_la = sum(beta_la_ls) / len(beta_la_ls)

    # Evaluate on test set
    err1 = Downstream_metric(beta, beta_ls, C_true, C_hat, r, x2[m:], y[m:])
    err2 = Downstream_metric(beta, beta_la, C_true, C_hat, r, x2[m:], y[m:])

    return err1, err2


if __name__ == '__main__':
    a = run_one_rep(0, 5000, 0.1, 100, 100, 100, 10, dvc="cuda")
