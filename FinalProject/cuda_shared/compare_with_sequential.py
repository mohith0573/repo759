#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA shared output matrices with sequential output matrices.")
    parser.add_argument("--seq-dir", default="../seq", help="Directory containing sequential_filter_*.csv files")
    parser.add_argument("--cout", type=int, required=True, help="Number of output filters")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for max absolute difference")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    cur_dir = Path(".")

    all_pass = True
    for co in range(args.cout):
        seq_file = seq_dir / f"sequential_filter_{co}.csv"
        shared_file = cur_dir / f"cuda_shared_filter_{co}.csv"

        if not seq_file.exists():
            print(f"Filter {co}: FAIL missing {seq_file}")
            all_pass = False
            continue
        if not shared_file.exists():
            print(f"Filter {co}: FAIL missing {shared_file}")
            all_pass = False
            continue

        seq = np.loadtxt(seq_file, delimiter=",")
        shared = np.loadtxt(shared_file, delimiter=",")

        if seq.shape != shared.shape:
            print(f"Filter {co}: FAIL shape mismatch seq={seq.shape}, cuda_shared={shared.shape}")
            all_pass = False
            continue

        diff = np.abs(seq - shared)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        status = "PASS" if max_diff <= args.tol else "FAIL"
        print(f"Filter {co}: {status} max_abs_diff={max_diff:.10e}, mean_abs_diff={mean_diff:.10e}")
        if max_diff > args.tol:
            all_pass = False

    print("FINAL RESULT:", "PASS" if all_pass else "FAIL")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
