"""
Training script for downstream linear regression with varying representation learning sample size (n).

This script runs multiple repetitions of downstream regression tasks with different
representation learning sample sizes, comparing OLS and Group Lasso estimators.
The downstream task training data size is fixed at m=10000.
"""

import numpy as np
import time
import torch
from Downstream_n import run_one_rep
import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", nargs='+', type=int, help="Dimension of raw data for three modality")
    parser.add_argument("--r", default=5, type=int, help="Dimension of latent vector")
    parser.add_argument("--c", default=0.1, type=float, help="noise level")
    parser.add_argument("--dvc", default='cuda', type=str, help="device used")
    return parser


if __name__ == "__main__":
    args = get_argparse().parse_args()
    d, r, c, dvc = args.d, args.r, args.c, args.dvc
    t0 = time.time()

    # Initialize error storage: (estimators, sample_sizes, repetitions, metrics)
    # 2 estimators: OLS, Group Lasso
    # 12 sample sizes (n: 500-8000), 100 repetitions, 9 metrics each
    err = torch.zeros((2, 12, 100, 9), device=dvc)
    for n_idx, n in enumerate([500, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]):
        for ii in range(100):
            results = run_one_rep(ii, n, c, d[0], d[1], d[2], r, dvc)

            err[0, n_idx, ii, :] = results[0]
            err[1, n_idx, ii, :] = results[1]
            np.save('Err_Linear/linear_n_d{}_r{}_c{}'.format(args.d, args.r, c), err.cpu().numpy())

        print(time.time() - t0)
