"""
Data generation module for representation learning experiments.
This module provides utilities to generate simulation data with hierarchical structure
for evaluating representation learning and downstream task performance.
"""

import random
import numpy as np
import torch


def W_generate(d1, d2, d3, r, device='cuda'):
    """
    Generate a hierarchical structured weight matrix on GPU.
    
    The matrix has three blocks corresponding to three modalities with dimensions d1, d2, d3.
    Each block has a specific sparse structure with r as the rank parameter.
    
    Args:
        d1: Dimension of the first modality
        d2: Dimension of the second modality
        d3: Dimension of the third modality
        r: Rank parameter controlling the width of structured components
        device: Device to place tensors ('cuda' or 'cpu')
    
    Returns:
        W: A (d1+d2+d3) x 7*r weight matrix with hierarchical structure
    """

    def ortho(rows, cols):
        """Generate a rows x cols orthogonal matrix using QR decomposition"""
        A = torch.randn(rows, cols, device=device)
        Q, _ = torch.linalg.qr(A)
        return Q

    def Full_rank(rows, cols):
        """
        Generate a rows x cols matrix with full column rank.
        The matrix is constructed as a product of three orthogonal-like matrices
        with a diagonal matrix in between for controlled singular values.
        """
        # Uniform sampling between 0.5 and 1.5 for singular values
        a = torch.rand(cols, device=device)+0.5
        a, _ = torch.sort(a, descending=True)
        res = ortho(rows, cols)@torch.diag(a)@ortho(cols, cols)
        return res

    # Construct hierarchical weight matrix by placing full-rank blocks
    # at specific positions to create sparse and modality-specific structure
    W = torch.zeros((d1+d2+d3, 7*r), device=device)
    
    # Generate 4 full-rank blocks for each modality
    W1 = [Full_rank(d1, r) for _ in range(4)]
    W2 = [Full_rank(d2, r) for _ in range(4)]
    W3 = [Full_rank(d3, r) for _ in range(4)]

    W[:d1, :3*r] = torch.cat([W1[0], W1[1], W1[2]], dim=1)
    W[:d1, 4*r:5*r] = W1[3]
    W[d1:(d1+d2), :2*r] = torch.cat([W2[0], W2[1]], dim=1)
    W[d1:(d1+d2), 3*r:4*r] = W2[2]
    W[d1:(d1+d2), 5*r: 6*r] = W2[3]
    W[-d3:, :r] = W3[0]
    W[-d3:, 2*r: 4*r] = torch.cat([W3[1], W3[2]], dim=1)
    W[-d3:, 6*r:7*r] = W3[3]

    return W


def beta_generate(r, device='cuda'):
    """
    Generate a sparse coefficient vector for linear combination of latent factors.
    
    Args:
        r: Rank parameter determining the length of the coefficient vector (7*r)
        device: Device to place tensors ('cuda' or 'cpu')
    
    Returns:
        beta: A normalized (unit norm) sparse coefficient vector of length 7*r
    """
    # Generate random coefficients and normalize to unit norm
    beta = torch.rand(7*r, device=device)
    
    # Optional: Set specific coefficients to zero for additional sparsity
    # beta[:2*r] = 0.      # Uncomment to zero out first 2*r elements
    # beta[6*r:] = 0.      # Uncomment to zero out last r elements

    return beta/torch.norm(beta)


def Data_generate(d1, d2, d3, r, n, c, device='cuda'):
    """
    Generate representation learning data with three modalities.
    
    This function generates observation data x by combining latent factors z
    with a weight matrix W, and adding Gaussian noise.
    
    Args:
        d1, d2, d3: Dimensions of the three modalities
        r: Rank parameter for the weight matrix
        n: Number of samples
        c: Noise level (variance parameter)
        device: Device to place tensors ('cuda' or 'cpu')
    
    Returns:
        x1: Observation matrix of shape (n, d1+d2+d3) with noise
        W: Weight matrix of shape (d1+d2+d3, 7*r)
    """
    # Generate latent representations (n samples, 7*r latent dimensions)
    z = torch.randn(n, 7*r, device=device)

    # Generate the hierarchical weight matrix
    W = W_generate(d1, d2, d3, r, device=device)
    
    # Add Gaussian noise with standard deviation sqrt(c)
    eps = torch.randn(n, d1+d2+d3, device=device) * torch.sqrt(torch.tensor(c, device=device))
    
    # Compute observations: x1 = z @ W.T + noise
    x1 = z @ W.T + eps

    return x1, W


def Label_generate(Weight, m, d1, d2, d3, r, c, cy, device='cuda'):
    """
    Generate linear regression data for downstream task learning.
    
    This function generates test/validation data with regression labels
    derived from the latent representations.
    
    Args:
        Weight: Pre-computed weight matrix from representation learning
        m: Number of samples
        d1, d2, d3: Dimensions of the three modalities
        r: Rank parameter
        c: Noise level for observations
        cy: Noise level for labels (output noise)
        device: Device to place tensors ('cuda' or 'cpu')
    
    Returns:
        x2: Observation matrix of shape (m, d1+d2+d3) with noise
        y: Regression labels of shape (m,)
        beta: Coefficient vector used for label generation
    """
    # Generate latent representations for new samples
    z2 = torch.randn(m, 7*r, device=device)
    
    # Add noise to observations
    eps2 = torch.randn(m, d1+d2+d3, device=device) * torch.sqrt(torch.tensor(c, device=device))
    x2 = z2 @ Weight.T + eps2

    beta = beta_generate(r, device=device)
    
    # Generate labels with noise: y = z @ beta + noise
    eps3 = torch.randn(m, device=device) * torch.sqrt(torch.tensor(cy, device=device))
    y = z2 @ beta + eps3

    return x2, y, beta


def generate_logistic_data(Weight, m, d, r, c, device='cuda'):
    """
    Generate binary classification data using logistic regression model.
    
    This function generates data with binary labels sampled from a Bernoulli
    distribution determined by sigmoid-transformed latent representations.
    
    Args:
        Weight: Pre-computed weight matrix from representation learning
        m: Number of samples
        d: Dimension of observations (3*d for internal use)
        r: Rank parameter
        c: Noise level for observations
        device: Device to place tensors ('cuda' or 'cpu')
    
    Returns:
        x2: Observation matrix of shape (m, 3*d) with noise
        y: Binary labels of shape (m,) sampled from Bernoulli distribution
        beta: Coefficient vector used for logit generation
    """
    # Generate latent representations for new samples
    z2 = torch.randn(m, 7*r, device=device)
    
    # Generate noisy observations
    eps2 = torch.randn(m, 3*d, device=device) * torch.sqrt(torch.tensor(c, device=device))
    x2 = z2 @ Weight.T + eps2

    # Generate label coefficients
    beta = beta_generate(r, device)
    
    # Generate binary labels via logistic function
    # Add noise (variance=6) to logits before applying sigmoid
    eps3 = torch.randn(m, device=device) * torch.sqrt(torch.tensor(6, device=device))
    logits = z2 @ beta + eps3
    
    # Convert logits to probabilities and sample binary labels
    prob = torch.sigmoid(logits)
    y = torch.bernoulli(prob)

    return x2, y, beta


def setup_seed(seed):
    """
    Set random seeds for reproducibility across multiple libraries.
    
    This function ensures deterministic behavior by seeding:
    - NumPy random number generator
    - Python's built-in random module
    - PyTorch CPU operations
    - PyTorch GPU operations (CUDA)
    - cuDNN backend (for deterministic GPU operations)
    
    Args:
        seed: Random seed value (integer)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False








