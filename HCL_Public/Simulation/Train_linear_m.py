"""
Training script for downstream linear regression with varying training sample size (m).

This script runs multiple repetitions of downstream regression tasks over different
training sample sizes, comparing OLS and Group Lasso estimators. The representation
learning data size is fixed at n=60.
"""

import numpy as np
import time
from Downstream_m import run_one_rep
import argparse
import torch

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
    # 10 sample sizes (m: 140-400), 100 repetitions, 9 metrics each
    err = torch.zeros((2, 10, 100, 9), device=dvc)
    for m_idx, m in enumerate([140, 160, 180, 200, 220, 240, 260, 280, 300, 400]):
        for ii in range(100):
            results = run_one_rep(ii, m, c, d[0], d[1], d[2], r, dvc)

            err[0, m_idx, ii, :] = results[0]
            err[1, m_idx, ii, :] = results[1]
            np.save('Err_Linear/linear_m_d{}_r{}_c{}'.format(args.d, args.r, c), err.cpu().numpy())
        print(time.time()-t0)
