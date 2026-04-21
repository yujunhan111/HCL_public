import numpy as np
import time
import torch
from HCL import run_one_rep
import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", nargs='+', type=int, help="Dimension of raw data for three modality")
    parser.add_argument("--r", default=10, type=int, help="Dimension of latent vector")
    parser.add_argument("--c", default=10, type=float, help="noise level")
    parser.add_argument("--dvc", default='cuda', type=str, help="device used")
    return parser


if __name__ == "__main__":
    args = get_argparse().parse_args()
    d, r, c, dvc = args.d, args.r, args.c, args.dvc
    t0 = time.time()

    err = torch.zeros((4, 13, 100, 13), device=dvc)
    for n_idx, n in enumerate([5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 25000, 30000]):
        for ii in range(100):
            results = run_one_rep(ii, n, c, d[0], d[1], d[2], r, dvc)

            err[0, n_idx, ii, :] = torch.tensor(results[0])
            err[1, n_idx, ii, :] = torch.tensor(results[1])
            err[2, n_idx, ii, :] = torch.tensor(results[2])
            err[3, n_idx, ii, :] = torch.tensor(results[3])
            np.save('Err_HCL/HCL_d{}_r{}_c{}'.format(args.d, args.r, c), err.cpu().numpy())

        print(time.time() - t0)

