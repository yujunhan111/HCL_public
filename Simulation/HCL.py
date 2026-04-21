import argparse
import time
from scipy.sparse.linalg import svds
from Data_generation import setup_seed, Data_generate
from scipy.linalg import orthogonal_procrustes
import torch
import numpy as np

t0 = time.time()
setup_seed(31)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default=1000, type=int, help="Dimension of raw data for one modality")
    parser.add_argument("--r", default=10, type=int, help="Dimension of latent vector")
    parser.add_argument("--dvc", default='cuda:0', type=str, help="device used")
    return parser


def pinv(S, r, device="cuda"):
    """Compute the pseudoinverse of a matrix."""
    S_np = S.cpu().numpy()
    U, s, Vt = svds(S_np, k=r, which='LM')
    idx = np.argsort(s)[::-1]
    S_inv = (U[:, idx] * (1 / s[idx])) @ Vt[idx, :]
    return torch.tensor(S_inv, device=device)


def Rec_W(S_sub, rank, device='cuda'):
    """Top-rank reconstruction using SVD on GPU."""
    S_np = S_sub.cpu().numpy()
    U, s, Vt = svds(S_np, k=rank, which='LM')
    idx = np.argsort(s)[::-1]
    W_rank = U[:, idx] * np.sqrt(s[idx])
    return torch.tensor(W_rank, device=device)


def HCL_grad(S, W_initial, mask, learning_rate=10 ** (-4)):
    V = W_initial.T
    loss_prev = float('inf')

    for i in range(10 ** 3):
        # compute loss and gradient
        diff = (V.T @V) - S
        loss = 0.5 * torch.sum(diff * diff)
        grad = -2 * V @ S + 2 * V @ V.T @ V
        grad = grad * mask

        # gradient descent
        V = V - learning_rate * grad

        if i > 0 and i % 10 == 0:
            learning_rate *= 0.1
        if i > 100 and abs(loss_prev - loss.item()) < 10 ** (-6):
            break
        loss_prev = loss.item()

    return V


def HCL_SVD(S, d1, d2, d3, r, device='cuda'):
    # individual structure
    idx13 = torch.cat([torch.arange(0, d1, device=device), torch.arange(d1+d2, d1+d2+d3, device=device)])
    S12 = S[:(d1+d2), :(d1+d2)]
    S13 = S[idx13][:, idx13]
    S23 = S[d1:, d1:] # 6r

    W1_1 = Rec_W(S[:d1, :d1] - S[:d1, d1:] @ pinv(S23, 6*r, device=device) @ S[:d1, d1:].T, r, device=device)
    W2_2 = Rec_W(S[d1:(d1+d2), d1:(d1+d2)] - S[d1:(d1+d2), idx13] @ pinv(S13, 6*r, device=device) @ S[d1:(d1+d2), idx13].T, r, device=device)
    W3_3 = Rec_W(S[-d3:, -d3:] - S[-d3:, :(d1+d2)] @ pinv(S12, 6*r, device=device) @ S[-d3:, :(d1+d2)].T, r, device=device)

    # partial structure
    S1 = S.clone()
    S1[:d1, :d1] -= W1_1@W1_1.T
    S1[d1:(d1+d2), d1:(d1+d2)] -= W2_2@W2_2.T
    S1[-d3:, -d3:] -= W3_3@W3_3.T # 3r

    W12_1 = Rec_W(S1[:d1, :d1] - S1[:d1, -d3:] @ pinv(S1[-d3:, -d3:], 3*r, device=device) @ S1[:d1, -d3:].T, r, device=device)
    W12_2 = Rec_W(S1[d1:(d1+d2), d1:(d1+d2)] - S1[d1:(d1+d2), -d3:] @ pinv(S1[-d3:, -d3:], 3*r, device=device) @ S1[d1:(d1+d2), -d3:].T, r, device=device)
    W13_1 = Rec_W(S1[:d1, :d1] - S1[:d1, d1:(d1+d2)] @ pinv(S1[d1:(d1+d2), d1:(d1+d2)], 3*r, device=device) @ S1[:d1, d1:(d1+d2)].T, r, device=device)
    W13_3 = Rec_W(S1[-d3:, -d3:] - S1[-d3:, d1:(d1+d2)] @ pinv(S1[d1:(d1+d2), d1:(d1+d2)], 3*r, device=device) @ S1[-d3:, d1:(d1+d2)].T, r, device=device)
    W23_2 = Rec_W(S1[d1:(d1+d2), d1:(d1+d2)] - S1[d1:(d1+d2), :d1] @ pinv(S1[:d1, :d1], 3*r, device=device) @ S1[d1:(d1+d2), :d1].T, r, device=device)
    W23_3 = Rec_W(S1[-d3:, -d3:] - S1[-d3:, :d1] @ pinv(S1[:d1, :d1], 3*r, device=device) @ S1[-d3:, :d1].T, r, device=device)

    # joint structure
    zero1 = torch.zeros((d1, r), device=device)
    zero2 = torch.zeros((d2, r), device=device)
    zero3 = torch.zeros((d3, r), device=device)
    W_par_row1 = torch.cat([W12_1, W13_1, zero1], dim=1)
    W_par_row2 = torch.cat([W12_2, zero2, W23_2], dim=1)
    W_par_row3 = torch.cat([zero3, W13_3, W23_3], dim=1)
    W_par = torch.cat([W_par_row1, W_par_row2, W_par_row3], dim=0)

    S2 = S1 - W_par @ W_par.T # r
    W123_1 = Rec_W(S2[:d1, :d1], r, device=device)
    W123_2 = Rec_W(S2[d1:(d1+d2), d1:(d1+d2)], r, device=device)
    W123_3 = Rec_W(S2[-d3:, -d3:], r, device=device)

    W_rec = torch.zeros((d1+d2+d3, 7*r), device=device)
    W_rec[:d1, :3*r] = torch.cat([W123_1, W12_1, W13_1], dim=1)
    W_rec[:d1, 4*r:5*r] = W1_1
    W_rec[d1:(d1+d2), :2*r] = torch.cat([W123_2, W12_2], dim=1)
    W_rec[d1:(d1+d2), 3*r:4*r] = W23_2
    W_rec[d1:(d1+d2), 5*r:6*r] = W2_2
    W_rec[-d3:, :r] = W123_3
    W_rec[-d3:, 2*r:4*r] = torch.cat([W13_3, W23_3], dim=1)
    W_rec[-d3:, 6*r:7*r] = W3_3
    return W_rec


def SLIDE(X, d1, d2, d3, r, tol=1e-6, max_iter=1000, device="cuda"):
    X_ls = [X[:, :d1], X[:, d1:(d1+d2)], X[:, -d3:]]
    S_ls = [np.r_[:3*r, 4*r:5*r], np.r_[:2*r, 3*r:4*r, 5*r:6*r], np.r_[:r, 2*r:4*r, 6*r:7*r]]

    # initialize U
    m = X.shape[0]
    A = torch.randn(m, 7*r, device=device)
    U = torch.linalg.qr(A)[0]

    # initialize V
    V = torch.zeros(d1+d2+d3, 7*r, device=device)
    V_ls = [V[:d1], V[d1:(d1+d2)], V[-d3:]]

    prev_UV = None

    for k in range(max_iter):
        # update V
        for i in range(3):
            V_ls[i][:, S_ls[i]] = X_ls[i].T@U[:, S_ls[i]]

        # update U
        XV = torch.zeros(m, 7*r, device=device)
        for i in range(3):
            XV += X_ls[i]@V_ls[i]

        R, L, Qh = torch.linalg.svd(XV, full_matrices=False)
        U_new = R@Qh
        UV = U_new@torch.cat(V_ls, dim=0).T

        if prev_UV is not None:
            diff = torch.norm(UV-prev_UV, p="fro")**2
            if diff<tol:
                U = U_new
                break

        prev_UV = UV.clone()
        U = U_new

    W = torch.zeros((7*r, d1+d2+d3), device=device)
    W_ls = [W[:, :d1], W[:, d1:(d1+d2)], W[:, -d3:]]
    for i in range(3):
        for j in range(7):
            R, L, Qh = torch.linalg.svd(U[:, j*r:(j+1)*r]@V_ls[i][:, j*r:(j+1)*r].T, full_matrices=False)
            W_ls[i][j*r:(j+1)*r, :] = L[:r]@Qh[:r, :]

    W = torch.cat(W_ls, dim=1)

    return W.T


def Sine_metric(W, W_rec, d1, d2, d3, r):
    W = W.cpu().numpy()
    W_rec = W_rec.cpu().numpy()
    ls = []

    R0, _ = orthogonal_procrustes(W_rec, W)
    W_rec = W_rec@R0

    W01 = W[:d1, np.r_[:3*r, 4*r:5*r]]
    W02 = W[d1:(d1+d2), np.r_[:2*r, 3*r:4*r, 5*r:6*r]]
    W03 = W[-d3:, np.r_[:r, 2*r:4*r, 6*r:7*r]]

    W0_rec1 = W_rec[:d1, np.r_[:3*r, 4*r:5*r]]
    W0_rec2 = W_rec[d1:(d1+d2), np.r_[:2*r, 3*r:4*r, 5*r:6*r]]
    W0_rec3 = W_rec[-d3:, np.r_[:r, 2*r:4*r, 6*r:7*r]]


    # structure based norm
    for i in range(3):
        for j in range(4):
            W0 = W01 if i == 0 else (W02 if i == 1 else W03)
            W0_rec = W0_rec1 if i == 0 else (W0_rec2 if i == 1 else W0_rec3)

            diff_ij = W0[:, j*r:(j+1)*r] @ W0[:, j*r:(j+1)*r].T - \
                      W0_rec[:, j*r:(j+1)*r] @ W0_rec[:, j*r:(j+1)*r].T
            ls.append(np.linalg.norm(diff_ij, ord='fro'))

    diff0 = W_rec@W_rec.T-W@W.T
    ls.append(np.linalg.norm(diff0, ord='fro'))

    return ls


def run_one_rep(ii, n, c, d1, d2, d3, r, dvc):
    setup_seed(20+ii)

    d = d1+d2+d3
    x1, W = Data_generate(d1, d2, d3, r=r, n=n, c=c, device=dvc)
    S_n = torch.cov(x1.T)  # sample covariance
    eigvals = torch.linalg.eigvalsh(S_n)  # ascending order
    sigma2_hat = eigvals[:(d-7*r)].mean()
    S_W = S_n - sigma2_hat * torch.eye(d, device=dvc)

    # mask matrix
    mask = np.ones((7*r, d), dtype=np.float32)
    mask[np.r_[3*r:4*r, 5*r:7*r], :d1] = 0
    mask[np.r_[2*r:3*r, 4*r:5*r, 6*r:7*r], d1:(d1+d2)] = 0
    mask[np.r_[r:2*r, 4*r:6*r], -d3:] = 0
    mask = torch.tensor(mask, device=dvc)

    # SVD_based method
    W_svd_naive = Rec_W(S_n, 7*r, device=dvc) # naive SVD
    W_svd = HCL_SVD(S_W, d1, d2,d3, r, device=dvc) # structured_based SVD

    # gradient_based method
    V_grad = HCL_grad(S_W, W_initial=W_svd, mask=mask)

    # SLIDE method
    W_slide = SLIDE(x1, d1, d2, d3, r, device=dvc)

    err_svd_naive = Sine_metric(W, W_svd_naive, d1, d2, d3, r)
    err_svd = Sine_metric(W, W_svd, d1, d2, d3, r)
    err_grad = Sine_metric(W, V_grad.T, d1, d2, d3, r)
    err_slide = Sine_metric(W, W_slide, d1, d2, d3, r)

    return err_svd_naive, err_svd, err_grad, err_slide

if __name__ == '__main__':
    a = run_one_rep(0, 5000, 6, 100, 500, 800, 3, dvc="cuda:0")
