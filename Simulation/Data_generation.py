import random
import numpy as np
import torch


def W_generate(d1, d2, d3, r, device='cuda'):
    """ generate a hierarchical structured matrix on GPU """

    def ortho(rows, cols):
        """ Generate a rows x cols orthogonal matrix """
        A = torch.randn(rows, cols, device=device)
        Q, _ = torch.linalg.qr(A)
        return Q

    def Full_rank(rows, cols):
        """ Generate a rows x cols matrix with full column rank """
        # Uniform sampling between 2 and 3
        a = torch.rand(cols, device=device)+0.5
        a, _ = torch.sort(a, descending=True)
        res = ortho(rows, cols)@torch.diag(a)@ortho(cols, cols)
        return res

    W = torch.zeros((d1+d2+d3, 7*r), device=device)
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
    """ generate a sparse coefficient on GPU """
    beta = torch.rand(7*r, device=device)
    # beta[:2*r] = 0.
    # beta[6*r:] = 0.

    return beta/torch.norm(beta)


def Data_generate(d1, d2, d3, r, n, c, device='cuda'):
    """ Generate representation learning data """
    # latent vector z
    z = torch.randn(n, 7*r, device=device)

    # coefficient matrix W
    W = W_generate(d1, d2, d3, r, device=device)
    eps = torch.randn(n, d1+d2+d3, device=device) * torch.sqrt(torch.tensor(c, device=device))
    x1 = z @ W.T + eps

    return x1, W


def Label_generate(Weight, m, d1, d2, d3, r, c, cy, device='cuda'):
    """ Generate linear regression data """
    z2 = torch.randn(m, 7*r, device=device)
    eps2 = torch.randn(m, d1+d2+d3, device=device) * torch.sqrt(torch.tensor(c, device=device))
    x2 = z2 @ Weight.T + eps2

    beta = beta_generate(r, device=device)
    eps3 = torch.randn(m, device=device) * torch.sqrt(torch.tensor(cy, device=device))
    y = z2 @ beta + eps3

    return x2, y, beta


def generate_logistic_data(Weight, m, d, r, c, device='cuda'):
    """ Generate logistic regression data """
    # covariates x
    z2 = torch.randn(m, 7*r, device=device)
    eps2 = torch.randn(m, 3*d, device=device) * torch.sqrt(torch.tensor(c, device=device))
    x2 = z2 @ Weight.T + eps2

    # label y
    beta = beta_generate(r, device)
    eps3 = torch.randn(m, device=device) * torch.sqrt(torch.tensor(6, device=device))
    logits = z2 @ beta + eps3
    prob = torch.sigmoid(logits)
    y = torch.bernoulli(prob)

    return x2, y, beta


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False








