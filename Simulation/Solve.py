import torch
import math


@torch.no_grad()
def group_prox(beta, groups, thresh, eps = 1e-12):
    """
    Compute the proximal operator for group Lasso regularization.

    Args:
       beta: torch.Tensor, coefficient vector to be processed.
       thresh: float, threshold parameter (step_size * lambda_m).

    Returns:
       out: torch.Tensor, coefficient vector after applying the proximal operator.
    """

    for g in groups:
        n = torch.linalg.norm(beta[g], ord=2)
        scale = torch.clamp(1.0 - thresh / (n + eps), min=0.0)
        beta[g] = scale * beta[g]

    return beta

@torch.no_grad()
def group_lasso_bcd(x, y, C_hat, sigma2_hat, lambda_m, groups, device, max_iter=2000, tol=1e-5):
    """
    Solve
        min_beta  0.5 * beta^T H beta - g^T beta
                  + lambda_m * sum_s ||beta_s||_2
    by cyclic block coordinate proximal descent.
    Update for block s:
        beta_s <- prox_{(lambda/L_s)||.||_2}( beta_s - grad_s / L_s )
    where L_s >= ||H_ss||_2.
    """
    m, r = x.shape
    H = (x.T@x)/m -sigma2_hat* (C_hat@C_hat.T)
    b = (x.T@y)/m
    beta = torch.zeros((r, 1), device=device)

    # Full gradient grad = H beta - b
    grad = H @ beta - b

    # Precompute group Lipschitz constants L_s = ||H_ss||_2
    L_groups = []
    for idx in groups:
        H_ss = H[idx][:, idx]
        eigs = torch.linalg.eigvalsh(H_ss)
        Ls = eigs.abs().max().clamp_min(1e-8)
        L_groups.append(Ls)

    for epoch in range(max_iter):
        max_rel = 0.0

        for s, idx in enumerate(groups):
            old = beta[idx].clone()
            Ls = L_groups[s]
            grad_s = grad[idx]

            # Majorized / proximal block step
            temp = old - grad_s / Ls
            new = group_prox(temp, [torch.arange(temp.numel(), device=temp.device)], lambda_m / Ls)

            # write back
            delta = new - old
            if torch.linalg.norm(delta) > 0:
                beta[idx] = new
                grad = grad + H[:, idx] @ delta

                rel = (torch.linalg.norm(delta) / torch.linalg.norm(old).clamp_min(1.0)).item()
                if rel > max_rel:
                    max_rel = rel

        if max_rel < tol:
            break

    return beta

@torch.no_grad()
def proximal_gradient_group_lasso(x, y, C_hat, sigma2_hat, lambda_m, groups,  device, max_iter=2000, tol=1e-5, lr=1e-3,):
    """
    Solve the optimization problem with group Lasso regularization using proximal gradient method.

    Objective function:
        (1/2m)||y - Phi beta||^2 + (1/2m) sigma2_hat * beta^T (C_hat C_hat^T) beta
        + lambda_m * sum_s ||beta_s||_2

    Args:
        x: torch.Tensor, design matrix of shape (m, r).
        y: torch.Tensor, target vector of shape (m,).
        C_hat: torch.Tensor, transformation matrix of shape (r, d).
        sigma2_hat: float, estimated noise variance.
        lambda_m: float, group Lasso regularization parameter.
        groups: list, each element is an index tensor defining groups in beta.
        max_iter: int, maximum number of iterations.
        tol: float, convergence tolerance.
        device: str, computing device ("cuda" or "cpu").

    Returns:
        beta: torch.Tensor, optimized coefficient vector of shape (r,).
    """

    m, r = x.shape
    H = (x.T@x)/m -sigma2_hat* (C_hat@C_hat.T)
    b = (x.T@y)/m

    if lr is None:
        L = torch.linalg.eigvalsh(H).abs().max().clamp_min(1e-8)
        lr = (1.0/L).item()

    beta = torch.zeros((r, 1), device=device)
    z = beta.clone()
    t = 1.0

    for k in range(max_iter):
        beta_old = beta.clone()
        grad = H @ z - b

        # gradient of smooth part
        beta = z - lr * grad
        # proximal step for group lasso
        beta = group_prox(beta, groups, thresh=lr * lambda_m)

        t_new = 0.5*(1+(1+4*t*t)**0.5)
        z = beta+((t-1)/t_new)*(beta-beta_old)
        t = t_new

        # stopping
        if torch.linalg.norm(beta - beta_old) < tol:
            break

    return beta

def compute_aic(y, x, beta, groups, eps=1e-12):
    n = y.shape[0]
    res = y - x @ beta
    rss = torch.sum(res ** 2)

    df = 0
    for g in groups:
        if torch.norm(beta[g], p=2) > 1e-8:
            df += len(g)

    aic = -2 * torch.log(rss + eps) + math.log(n) * df
    return aic.item()

def make_folds(m, n_folds=3, seed=0):
    """
    Split data into folds for cross-validation.

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