#!/usr/bin/env python3
"""Compare Python reference output matrices with sequential output matrices."""

import argparse
from pathlib import Path


def read_matrix(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",") if x.strip()])
    return rows


def compare(a, b):
    if len(a) != len(b):
        raise ValueError(f"row count mismatch: {len(a)} vs {len(b)}")
    max_diff = 0.0
    mean_sum = 0.0
    count = 0
    for r1, r2 in zip(a, b):
        if len(r1) != len(r2):
            raise ValueError(f"column count mismatch: {len(r1)} vs {len(r2)}")
        for x, y in zip(r1, r2):
            d = abs(x - y)
            max_diff = max(max_diff, d)
            mean_sum += d
            count += 1
    mean_diff = mean_sum / count if count else 0.0
    return max_diff, mean_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", default="../seq", help="Path to sequential directory")
    parser.add_argument("--cout", type=int, required=True, help="Number of output filters")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    ok = True

    for co in range(args.cout):
        seq_file = seq_dir / f"sequential_filter_{co}.csv"
        py_file = Path(f"python_reference_filter_{co}.csv")

        if not seq_file.exists():
            print(f"Filter {co}: FAIL missing {seq_file}")
            ok = False
            continue
        if not py_file.exists():
            print(f"Filter {co}: FAIL missing {py_file}")
            ok = False
            continue

        seq = read_matrix(seq_file)
        py = read_matrix(py_file)
        max_diff, mean_diff = compare(seq, py)
        status = "PASS" if max_diff <= args.tol else "FAIL"
        print(f"Filter {co}: {status} max_abs_diff={max_diff:.10e} mean_abs_diff={mean_diff:.10e}")
        if max_diff > args.tol:
            ok = False

    print("FINAL RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
