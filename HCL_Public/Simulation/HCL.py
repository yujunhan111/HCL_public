"""
Hierarchical Contrastive Learning (HCL) module for multimodal representation learning.

This module implements methods for learning hierarchical structure in multimodal data:
- HCL_SVD: Structure-based SVD decomposition using hierarchical masking
- HCL_grad: Gradient-based optimization with masked constraints
- SLIDE: Subspace Learning with Incomplete Data Exploration
- Sine_metric: Angular similarity metric for evaluating learned representations
"""

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
    """Compute the rank-r pseudoinverse of a matrix using truncated SVD.
    
    Args:
        S: torch.Tensor, input matrix to invert
        r: int, rank of the truncated SVD
        device: str, device for output tensor
    
    Returns:
        S_inv: torch.Tensor, rank-r pseudoinverse of S
    """
    S_np = S.cpu().numpy()
    U, s, Vt = svds(S_np, k=r, which='LM')
    idx = np.argsort(s)[::-1]
    S_inv = (U[:, idx] * (1 / s[idx])) @ Vt[idx, :]
    return torch.tensor(S_inv, device=device)


def Rec_W(S_sub, rank, device='cuda'):
    """Reconstruct weight matrix using top-rank SVD decomposition.
    
    Extracts the top-rank components and reconstructs as W = U * sqrt(S).
    
    Args:
        S_sub: torch.Tensor, input matrix for decomposition
        rank: int, number of top singular components to keep
        device: str, device for output tensor
    
    Returns:
        W_rank: torch.Tensor, reconstructed weight matrix of shape (m, rank)
    """
    S_np = S_sub.cpu().numpy()
    U, s, Vt = svds(S_np, k=rank, which='LM')
    idx = np.argsort(s)[::-1]
    W_rank = U[:, idx] * np.sqrt(s[idx])
    return torch.tensor(W_rank, device=device)


def HCL_grad(S, W_initial, mask, learning_rate=10 ** (-4)):
    """Optimize weight matrix using masked gradient descent.
    
    Objective: minimize ||V^T V - S||_F^2 subject to mask constraints.
    The mask matrix specifies which elements should contribute to the objective.
    
    Args:
        S: torch.Tensor, target covariance or Gram matrix
        W_initial: torch.Tensor, initial weight matrix (shape: d x 7r)
        mask: torch.Tensor, binary mask for gradient (shape: 7r x d)
        learning_rate: float, initial learning rate for gradient descent
    
    Returns:
        V: torch.Tensor, optimized matrix (7r x d)
    """
    # Initialize optimization variable
    V = W_initial.T
    loss_prev = float('inf')

    for i in range(10 ** 3):
        # Compute reconstruction error and loss function
        diff = (V.T @ V) - S
        loss = 0.5 * torch.sum(diff * diff)
        
        # Compute gradient of objective function
        # grad = 2V(V^T V - S) after simplification
        grad = -2 * V @ S + 2 * V @ V.T @ V
        # Apply mask: zero out gradients for non-active components
        grad = grad * mask

        # Gradient descent update
        V = V - learning_rate * grad

        # Learning rate decay every 10 iterations
        if i > 0 and i % 10 == 0:
            learning_rate *= 0.1
        
        # Early stopping: check for convergence
        if i > 100 and abs(loss_prev - loss.item()) < 10 ** (-6):
            break
        loss_prev = loss.item()

    return V


def HCL_SVD(S, d1, d2, d3, r, device='cuda'):
    """Hierarchical Contrastive Learning using structure-based SVD decomposition.
    
    Decomposes the weight matrix into three hierarchies:
    1. Individual structure: r-rank components specific to each modality
    2. Pairwise structure: r-rank components shared by pairs of modalities
    3. Joint structure: r-rank components shared by all three modalities
    
    Args:
        S: torch.Tensor, estimated latent covariance matrix (after removing noise)
        d1, d2, d3: int, dimensions of three modalities
        r: int, rank of individual/pairwise/joint components
        device: str, device for computations
    
    Returns:
        W_rec: torch.Tensor, reconstructed weight matrix of shape (d1+d2+d3, 7*r)
    """
    # Extract modality-specific covariance blocks
    idx13 = torch.cat([torch.arange(0, d1, device=device), torch.arange(d1+d2, d1+d2+d3, device=device)])
    # Extract modality-specific covariance blocks
    idx13 = torch.cat([torch.arange(0, d1, device=device), torch.arange(d1+d2, d1+d2+d3, device=device)])
    S12 = S[:(d1+d2), :(d1+d2)]  # Covariance of modalities 1 and 2
    S13 = S[idx13][:, idx13]      # Covariance of modalities 1 and 3
    S23 = S[d1:, d1:]              # Covariance of modalities 2 and 3 (6r components)

    # Step 1: Extract individual (modality-specific) r-rank components
    # Use Schur complement to remove contributions from other modalities
    W1_1 = Rec_W(S[:d1, :d1] - S[:d1, d1:] @ pinv(S23, 6*r, device=device) @ S[:d1, d1:].T, r, device=device)
    W2_2 = Rec_W(S[d1:(d1+d2), d1:(d1+d2)] - S[d1:(d1+d2), idx13] @ pinv(S13, 6*r, device=device) @ S[d1:(d1+d2), idx13].T, r, device=device)
    W3_3 = Rec_W(S[-d3:, -d3:] - S[-d3:, :(d1+d2)] @ pinv(S12, 6*r, device=device) @ S[-d3:, :(d1+d2)].T, r, device=device)

    # Step 2: Remove individual components and extract pairwise r-rank components
    S1 = S.clone()
    S1[:d1, :d1] -= W1_1@W1_1.T
    S1[d1:(d1+d2), d1:(d1+d2)] -= W2_2@W2_2.T
    S1[-d3:, -d3:] -= W3_3@W3_3.T  # Remaining variance is 3r per modality

    # Extract pairwise components: 1-2, 1-3, 2-3 interactions
    W12_1 = Rec_W(S1[:d1, :d1] - S1[:d1, -d3:] @ pinv(S1[-d3:, -d3:], 3*r, device=device) @ S1[:d1, -d3:].T, r, device=device)
    W12_2 = Rec_W(S1[d1:(d1+d2), d1:(d1+d2)] - S1[d1:(d1+d2), -d3:] @ pinv(S1[-d3:, -d3:], 3*r, device=device) @ S1[d1:(d1+d2), -d3:].T, r, device=device)
    W13_1 = Rec_W(S1[:d1, :d1] - S1[:d1, d1:(d1+d2)] @ pinv(S1[d1:(d1+d2), d1:(d1+d2)], 3*r, device=device) @ S1[:d1, d1:(d1+d2)].T, r, device=device)
    W13_3 = Rec_W(S1[-d3:, -d3:] - S1[-d3:, d1:(d1+d2)] @ pinv(S1[d1:(d1+d2), d1:(d1+d2)], 3*r, device=device) @ S1[-d3:, d1:(d1+d2)].T, r, device=device)
    W23_2 = Rec_W(S1[d1:(d1+d2), d1:(d1+d2)] - S1[d1:(d1+d2), :d1] @ pinv(S1[:d1, :d1], 3*r, device=device) @ S1[d1:(d1+d2), :d1].T, r, device=device)
    W23_3 = Rec_W(S1[-d3:, -d3:] - S1[-d3:, :d1] @ pinv(S1[:d1, :d1], 3*r, device=device) @ S1[-d3:, :d1].T, r, device=device)

    # Step 3: Construct pairwise component matrix
    # Each pairwise component has r dimensions, placed in 3 specific columns
    zero1 = torch.zeros((d1, r), device=device)
    zero2 = torch.zeros((d2, r), device=device)
    zero3 = torch.zeros((d3, r), device=device)
    W_par_row1 = torch.cat([W12_1, W13_1, zero1], dim=1)
    W_par_row2 = torch.cat([W12_2, zero2, W23_2], dim=1)
    W_par_row3 = torch.cat([zero3, W13_3, W23_3], dim=1)
    W_par = torch.cat([W_par_row1, W_par_row2, W_par_row3], dim=0)

    # Step 4: Extract joint r-rank components (shared by all three modalities)
    S2 = S1 - W_par @ W_par.T  # Remove pairwise contributions
    W123_1 = Rec_W(S2[:d1, :d1], r, device=device)
    W123_2 = Rec_W(S2[d1:(d1+d2), d1:(d1+d2)], r, device=device)
    W123_3 = Rec_W(S2[-d3:, -d3:], r, device=device)

    # Step 5: Construct the full weight matrix with 7*r columns
    # Column structure: [0:r]=joint, [r:2r]=W13, [2r:3r]=W23, [3r:4r]=W12, 
    #                   [4r:5r]=indiv1, [5r:6r]=indiv2, [6r:7r]=indiv3
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
    """Subspace Learning with Incomplete Data Exploration (SLIDE).
    
    Learns the latent representation by optimizing the low-rank subspace
    that explains the variance in multimodal data, respecting modality-specific structures.
    
    Args:
        X: torch.Tensor, input data matrix of shape (n, d1+d2+d3)
        d1, d2, d3: int, dimensions of three modalities
        r: int, rank of latent subspace
        tol: float, convergence tolerance
        max_iter: int, maximum iterations
        device: str, computation device
    
    Returns:
        W.T: torch.Tensor, transposed weight matrix of shape (d1+d2+d3, 7*r)
    """
    # Separate data by modality
    X_ls = [X[:, :d1], X[:, d1:(d1+d2)], X[:, -d3:]]
    S_ls = [np.r_[:3*r, 4*r:5*r], np.r_[:2*r, 3*r:4*r, 5*r:6*r], np.r_[:r, 2*r:4*r, 6*r:7*r]]

    # Initialize U (left singular vectors) using QR decomposition
    m = X.shape[0]
    A = torch.randn(m, 7*r, device=device)
    U = torch.linalg.qr(A)[0]

    # Initialize V (right singular vectors / weight matrix)
    V = torch.zeros(d1+d2+d3, 7*r, device=device)
    V_ls = [V[:d1], V[d1:(d1+d2)], V[-d3:]]

    prev_UV = None

    for k in range(max_iter):
        # Update V: Extract relevant columns based on structure
        for i in range(3):
            V_ls[i][:, S_ls[i]] = X_ls[i].T@U[:, S_ls[i]]

        # Update U: Perform SVD on the product X*V
        XV = torch.zeros(m, 7*r, device=device)
        for i in range(3):
            XV += X_ls[i]@V_ls[i]

        R, L, Qh = torch.linalg.svd(XV, full_matrices=False)
        U_new = R@Qh
        
        # Compute reconstruction error for convergence check
        UV = U_new@torch.cat(V_ls, dim=0).T

        if prev_UV is not None:
            diff = torch.norm(UV-prev_UV, p="fro")**2
            if diff<tol:
                U = U_new
                break

        prev_UV = UV.clone()
        U = U_new

    # Reconstruct W by extracting singular components for each factor
    W = torch.zeros((7*r, d1+d2+d3), device=device)
    W_ls = [W[:, :d1], W[:, d1:(d1+d2)], W[:, -d3:]]
    for i in range(3):
        for j in range(7):
            R, L, Qh = torch.linalg.svd(U[:, j*r:(j+1)*r]@V_ls[i][:, j*r:(j+1)*r].T, full_matrices=False)
            W_ls[i][j*r:(j+1)*r, :] = L[:r]@Qh[:r, :]

    W = torch.cat(W_ls, dim=1)

    return W.T


def Sine_metric(W, W_rec, d1, d2, d3, r):
    """Compute angular distance metrics for evaluating weight matrix recovery quality.
    
    Uses principal angles (sine distances) between true and recovered subspaces
    to measure learning performance across different structure levels.
    
    Args:
        W: torch.Tensor, true weight matrix (d x 7r)
        W_rec: torch.Tensor, recovered weight matrix (d x 7r)
        d1, d2, d3: int, dimensions of three modalities
        r: int, rank of components
    
    Returns:
        ls: list, Frobenius norm distances for each component (13 metrics total:
            - 4 components for modality 1
            - 4 components for modality 2 
            - 4 components for modality 3
            - 1 overall Gramian difference)
    """
    # Convert to numpy for processing
    W = W.cpu().numpy()
    W_rec = W_rec.cpu().numpy()
    ls = []

    # Align recovered matrix to true matrix using orthogonal Procrustes
    R0, _ = orthogonal_procrustes(W_rec, W)
    W_rec = W_rec@R0

    W01 = W[:d1, np.r_[:3*r, 4*r:5*r]]
    W02 = W[d1:(d1+d2), np.r_[:2*r, 3*r:4*r, 5*r:6*r]]
    W03 = W[-d3:, np.r_[:r, 2*r:4*r, 6*r:7*r]]

    W0_rec1 = W_rec[:d1, np.r_[:3*r, 4*r:5*r]]
    W0_rec2 = W_rec[d1:(d1+d2), np.r_[:2*r, 3*r:4*r, 5*r:6*r]]
    W0_rec3 = W_rec[-d3:, np.r_[:r, 2*r:4*r, 6*r:7*r]]


    # Compute Frobenius norm distances for structure-specific Gram matrices
    # This evaluates how well the principal angles are recovered
    for i in range(3):
        for j in range(4):
            # Select current modality's data
            W0 = W01 if i == 0 else (W02 if i == 1 else W03)
            W0_rec = W0_rec1 if i == 0 else (W0_rec2 if i == 1 else W0_rec3)

            diff_ij = W0[:, j*r:(j+1)*r] @ W0[:, j*r:(j+1)*r].T - \
                      W0_rec[:, j*r:(j+1)*r] @ W0_rec[:, j*r:(j+1)*r].T
            ls.append(np.linalg.norm(diff_ij, ord='fro'))

    diff0 = W_rec@W_rec.T-W@W.T
    ls.append(np.linalg.norm(diff0, ord='fro'))

    return ls


def run_one_rep(ii, n, c, d1, d2, d3, r, dvc):
    """Run one repetition of the HCL learning experiment.
    
    Generates synthetic data, learns the hierarchical weight matrix using multiple methods,
    and evaluates recovery quality using angular distance metrics.
    
    Args:
        ii: int, repetition index (used for seeding)
        n: int, number of samples
        c: float, observation noise level
        d1, d2, d3: int, dimensions of three modalities
        r: int, rank of latent components
        dvc: str, device for computation
    
    Returns:
        (err_svd_naive, err_svd, err_grad, err_slide): tuple of error metrics for each method
    """
    # Set random seed for reproducibility
    setup_seed(20+ii)

    # Generate synthetic data with hierarchical structure
    d = d1+d2+d3
    x1, W = Data_generate(d1, d2, d3, r=r, n=n, c=c, device=dvc)
    S_n = torch.cov(x1.T)  # sample covariance
    eigvals = torch.linalg.eigvalsh(S_n)  # ascending order
    sigma2_hat = eigvals[:(d-7*r)].mean()
    S_W = S_n - sigma2_hat * torch.eye(d, device=dvc)

    # Create mask matrix to encode hierarchical structure constraints
    mask = np.ones((7*r, d), dtype=np.float32)
    mask[np.r_[3*r:4*r, 5*r:7*r], :d1] = 0
    mask[np.r_[2*r:3*r, 4*r:5*r, 6*r:7*r], d1:(d1+d2)] = 0
    mask[np.r_[r:2*r, 4*r:6*r], -d3:] = 0
    mask = torch.tensor(mask, device=dvc)

    # Method 1: Naive SVD (baseline - no structure)
    W_svd_naive = Rec_W(S_n, 7*r, device=dvc)
    
    # Method 2: Structure-based SVD (hierarchical decomposition)
    W_svd = HCL_SVD(S_W, d1, d2, d3, r, device=dvc)

    # Method 3: Gradient-based optimization with masked constraints
    V_grad = HCL_grad(S_W, W_initial=W_svd, mask=mask)

    # Method 4: SLIDE (Subspace Learning with Incomplete Data Exploration)
    W_slide = SLIDE(x1, d1, d2, d3, r, device=dvc)

    # Evaluate all methods using angular distance metrics
    err_svd_naive = Sine_metric(W, W_svd_naive, d1, d2, d3, r)
    err_svd = Sine_metric(W, W_svd, d1, d2, d3, r)
    err_grad = Sine_metric(W, V_grad.T, d1, d2, d3, r)
    err_slide = Sine_metric(W, W_slide, d1, d2, d3, r)

    return err_svd_naive, err_svd, err_grad, err_slide

if __name__ == '__main__':
    a = run_one_rep(0, 5000, 6, 100, 500, 800, 3, dvc="cuda:0")
