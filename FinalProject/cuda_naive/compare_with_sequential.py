import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA naive output matrices with sequential output matrices.")
    parser.add_argument("--seq-dir", default="../seq", help="Path to sequential directory")
    parser.add_argument("--cuda-dir", default=".", help="Path to CUDA naive directory")
    parser.add_argument("--cout", type=int, required=True, help="Number of output filters")
    parser.add_argument("--tol", type=float, default=1e-4, help="Absolute tolerance")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    cuda_dir = Path(args.cuda_dir)

    all_pass = True

    for co in range(args.cout):
        seq_file = seq_dir / f"sequential_filter_{co}.csv"
        cuda_file = cuda_dir / f"cuda_naive_filter_{co}.csv"

        if not seq_file.exists():
            print(f"Missing sequential file: {seq_file}")
            all_pass = False
            continue
        if not cuda_file.exists():
            print(f"Missing CUDA naive file: {cuda_file}")
            all_pass = False
            continue

        seq = np.loadtxt(seq_file, delimiter=",")
        cuda = np.loadtxt(cuda_file, delimiter=",")

        if seq.shape != cuda.shape:
            print(f"Filter {co}: FAIL shape mismatch seq={seq.shape}, cuda={cuda.shape}")
            all_pass = False
            continue

        diff = np.abs(seq - cuda)
        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))
        status = "PASS" if max_abs_diff <= args.tol else "FAIL"

        print(f"Filter {co}: {status} max_abs_diff={max_abs_diff:.10e}, mean_abs_diff={mean_abs_diff:.10e}")

        if max_abs_diff > args.tol:
            all_pass = False

    print("FINAL RESULT:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
