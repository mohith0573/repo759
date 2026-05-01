#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def load_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.loadtxt(path, delimiter=",")


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA shared output matrices with sequential output matrices.")
    parser.add_argument("--seq-dir", type=str, default="../seq", help="Directory containing sequential_filter_*.csv")
    parser.add_argument("--cuda-dir", type=str, default=".", help="Directory containing cuda_shared_filter_*.csv")
    parser.add_argument("--cout", type=int, required=True, help="Number of output filters")
    parser.add_argument("--tol", type=float, default=1e-4, help="Absolute tolerance")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    cuda_dir = Path(args.cuda_dir)

    overall_pass = True

    for co in range(args.cout):
        seq_file = seq_dir / f"sequential_filter_{co}.csv"
        cuda_file = cuda_dir / f"cuda_shared_filter_{co}.csv"

        seq = load_matrix(seq_file)
        cuda = load_matrix(cuda_file)

        if seq.shape != cuda.shape:
            print(f"Filter {co}: FAIL shape mismatch seq={seq.shape}, cuda_shared={cuda.shape}")
            overall_pass = False
            continue

        abs_diff = np.abs(seq - cuda)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        status = "PASS" if max_diff <= args.tol else "FAIL"

        print(f"Filter {co}: {status}")
        print(f"  max_abs_diff  = {max_diff:.10e}")
        print(f"  mean_abs_diff = {mean_diff:.10e}")

        if max_diff > args.tol:
            overall_pass = False

    print()
    print("FINAL RESULT:", "PASS" if overall_pass else "FAIL")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
