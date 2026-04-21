import argparse
from Data_generation import setup_seed, Data_generate, Label_generate
import numpy as np
from scipy.linalg import orthogonal_procrustes
import torch
from HCL import HCL_SVD, HCL_grad
from Solve import group_lasso_bcd, proximal_gradient_group_lasso

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default=1000, type=int, help="Dimension of raw data for one modality")
    parser.add_argument("--r", default=10, type=int, help="Dimension of latent vector")
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
    Perform K-fold cross-validation to select the optimal group Lasso regularization parameter and return the corresponding coefficient estimate.

    Args:
        x: torch.Tensor, design matrix of shape (m, r).
        y: torch.Tensor, target vector of shape (m,).
        C_hat: torch.Tensor, transformation matrix of shape (r, d).
        sigma2_hat: float, estimated noise variance.
        groups: list, each element is an index tensor defining groups in beta.
        device: str, computing device ("cuda" or "cpu").

    Returns:
        best_beta: torch.Tensor, coefficient estimate with the optimal regularization parameter of shape (r,).
    """

    m = x.shape[0]
    folds = make_folds(m, n_folds=3)
    cv_errors = []
    lambda_list = [10**i for i in np.arange(-6,1, dtype=np.float32)]
    for alpha in lambda_list:
        fold_scores = []

        for k in range(3):
            val_idx = folds[k]
            train_idx = torch.cat([folds[i] for i in range(3) if i != k])
            x_train, y_train, x_val, y_val = x[train_idx], y[train_idx], x[val_idx], y[val_idx]
            beta_hat = group_lasso_bcd(x, y, C_hat, sigma2_hat, alpha, groups, device=device)

            # Predict and score
            residual = x_val@beta_hat-y_val
            score = (residual.T@residual)/y_val.shape[0]
            fold_scores.append(score.reshape(-1).item())

        cv_errors.append(np.mean(fold_scores))

    best_index = np.argmin(cv_errors)
    best_lambda = lambda_list[best_index]
    best_beta = group_lasso_bcd(x, y, C_hat, sigma2_hat, best_lambda, groups, device=device)

    return best_beta.reshape(-1, 1)


def Downstream_metric(beta, beta_hat, C, C_hat, r, x, y):
    Risk = torch.mean((y - x@C_hat.T@beta_hat)**2) - torch.mean((y - x@C.T@beta)**2)
    dis = torch.linalg.norm(beta_hat-beta, 2)
    dis_g = [torch.linalg.norm(beta_hat[i*r:(i+1)*r]-beta[i*r:(i+1)*r], 2) for i in range(7)]
    res = dis_g+[dis, Risk]

    return torch.tensor(res)


def run_one_rep(ii, n, c, d1, d2, d3, r, dvc):
    setup_seed(20+ii)

    d = d1+d2+d3
    m = 10000
    x1, W = Data_generate(d1, d2, d3, r=r, n=n, c=c, device=dvc)
    x2, y, beta = Label_generate(W, m=2*m, d1=d1, d2=d2, d3=d3, r=r, c=c, cy=0.1, device=dvc)
    y, beta = y.reshape(-1, 1), beta.reshape(-1, 1)
    S_n = torch.cov(x1.T)  # sample covariance
    eigvals = torch.linalg.eigvalsh(S_n)  # ascending order
    sigma2_hat = eigvals[:(d-7*r)].mean()
    S_W = S_n-sigma2_hat*torch.eye(d, device=dvc)

    # mask matrix
    mask = np.ones((7*r, d), dtype=np.float32)
    mask[np.r_[3*r:4*r, 5*r:7*r], :d1] = 0
    mask[np.r_[2*r:3*r, 4*r:5*r, 6*r:7*r], d1:(d1+d2)] = 0
    mask[np.r_[r:2*r, 4*r:6*r], -d3:] = 0
    mask = torch.tensor(mask, device=dvc)

    # contrastive learning-SVD
    W_ls = []
    for _ in range(20):
        W_svd = HCL_SVD(S_W, d1, d2, d3, r, device=dvc)
        V_grad = HCL_grad(S_W, W_initial=W_svd, mask=mask)
        R = orthogonal_procrustes(V_grad.cpu().numpy().T, W.cpu().numpy())[0]
        R = torch.tensor(R, device=dvc)
        W_grad = V_grad.T@R
        W_ls.append(W_grad)
    W_grad = sum(W_ls) / len(W_ls)

    # downstream task
    C_true = torch.linalg.inv(W.T@W)@W.T
    C_hat = torch.linalg.inv(W_grad.T@W_grad)@W_grad.T
    X2 = x2[:m] @ C_hat.T

    # OLS estimator
    A = (X2.T@X2)/m-sigma2_hat*(C_hat@C_hat.T)
    beta_ls = torch.linalg.solve(A, (X2.T@y[:m])/m)

    # Group Lasso estimator
    groups = [list(range(i*r, (i+1)*r)) for i in range(7)]
    beta_la_ls = []
    for _ in range(20):
        beta_la = GL_estimator(X2, y[:m], C_hat, sigma2_hat, groups, device=dvc)
        beta_la_ls.append(beta_la)
    beta_la = sum(beta_la_ls) / len(beta_la_ls)

    err1 = Downstream_metric(beta, beta_ls, C_true, C_hat, r, x2[m:], y[m:])
    err2 = Downstream_metric(beta, beta_la, C_true, C_hat, r, x2[m:], y[m:])

    return err1, err2


if __name__ == '__main__':
    a = run_one_rep(0, 5000, 0.1, 100, 100, 100, 10, dvc="cuda")
