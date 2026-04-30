#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def load_matrix(path):
    return np.loadtxt(path, delimiter=',')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cout', type=int, default=8)
    parser.add_argument('--seq-dir', type=str, default='../seq')
    parser.add_argument('--tol', type=float, default=1e-4)
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    ok = True

    for co in range(args.cout):
        seq_file = seq_dir / f'sequential_filter_{co}.csv'
        omp_file = Path(f'openmp_filter_{co}.csv')

        if not seq_file.exists() or not omp_file.exists():
            print(f'Filter {co}: MISSING FILE')
            print(f'  seq: {seq_file}')
            print(f'  omp: {omp_file}')
            ok = False
            continue

        seq = load_matrix(seq_file)
        omp = load_matrix(omp_file)
        max_abs_diff = float(np.max(np.abs(seq - omp)))
        status = 'PASS' if max_abs_diff <= args.tol else 'FAIL'
        print(f'Filter {co}: {status}, max_abs_diff={max_abs_diff:.8e}')
        if status == 'FAIL':
            ok = False

    print('FINAL RESULT:', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
