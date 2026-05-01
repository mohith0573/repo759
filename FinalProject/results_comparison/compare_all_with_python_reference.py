#!/usr/bin/env python3
"""
Compare output filter matrices from all implementations against the Python reference.

Expected project layout:

Project/
├── sequential_implementation/
│   ├── input.csv
│   ├── kernel.csv
│   ├── sequential_filter_0.csv
│   └── ...
├── openmp/
│   ├── input.csv
│   ├── kernel.csv
│   ├── openmp_filter_0.csv
│   └── ...
├── cuda_naive/
│   ├── input.csv
│   ├── kernel.csv
│   ├── cuda_naive_filter_0.csv
│   └── ...
├── cuda_shared/
│   ├── input.csv
│   ├── kernel.csv
│   ├── cuda_shared_filter_0.csv
│   └── ...
├── python_ref/
│   ├── input.csv
│   ├── kernel.csv
│   ├── python_reference_filter_0.csv
│   └── ...
└── results_comparison/
    ├── compare_all_with_python_reference.py
    ├── comparison_job.sh
    └── ...

This script should be run from inside results_comparison/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_matrix_csv(path: Path) -> List[List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing matrix file: {path}")

    matrix: List[List[float]] = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row_id, row in enumerate(reader):
            if not row:
                continue
            try:
                matrix.append([float(x.strip()) for x in row if x.strip() != ""])
            except ValueError as exc:
                raise ValueError(f"Could not parse numeric values in {path} at row {row_id}") from exc

    if not matrix:
        raise ValueError(f"Matrix file is empty: {path}")

    width = len(matrix[0])
    for r, row in enumerate(matrix):
        if len(row) != width:
            raise ValueError(
                f"Non-rectangular matrix in {path}: row 0 has {width} columns, row {r} has {len(row)}"
            )

    return matrix


def compare_matrices(ref: List[List[float]], test: List[List[float]]) -> Tuple[float, float, float, int, int]:
    if len(ref) != len(test) or len(ref[0]) != len(test[0]):
        raise ValueError(
            f"Shape mismatch: reference is {len(ref)}x{len(ref[0])}, "
            f"test is {len(test)}x{len(test[0])}"
        )

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    sum_sq_diff = 0.0
    compared = 0

    for r in range(len(ref)):
        for c in range(len(ref[0])):
            d = abs(ref[r][c] - test[r][c])
            max_abs_diff = max(max_abs_diff, d)
            sum_abs_diff += d
            sum_sq_diff += d * d
            compared += 1

    mean_abs_diff = sum_abs_diff / compared if compared else 0.0
    rmse = math.sqrt(sum_sq_diff / compared) if compared else 0.0
    return max_abs_diff, mean_abs_diff, rmse, compared, len(ref) * len(ref[0])


def count_exceeding(ref: List[List[float]], test: List[List[float]], tol: float) -> int:
    count = 0
    for r in range(len(ref)):
        for c in range(len(ref[0])):
            if abs(ref[r][c] - test[r][c]) > tol:
                count += 1
    return count


def check_input_kernel_files(method_dirs: dict, reference_method: str, output_csv: Path) -> bool:
    """
    Checks whether input.csv and kernel.csv are byte-identical across all folders.
    """
    rows = []
    all_ok = True

    ref_dir = method_dirs[reference_method]

    for filename in ["input.csv", "kernel.csv"]:
        ref_file = ref_dir / filename
        if not ref_file.exists():
            rows.append([filename, reference_method, str(ref_file), "MISSING", "", "FAIL"])
            all_ok = False
            continue

        ref_hash = sha256_file(ref_file)

        for method, d in method_dirs.items():
            file_path = d / filename
            if not file_path.exists():
                rows.append([filename, method, str(file_path), "MISSING", "", "FAIL"])
                all_ok = False
                continue

            h = sha256_file(file_path)
            status = "PASS" if h == ref_hash else "FAIL"
            if status == "FAIL":
                all_ok = False

            rows.append([filename, method, str(file_path), h, ref_hash, status])

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "method", "path", "sha256", "reference_sha256", "status"])
        writer.writerows(rows)

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare all implementation output matrices against Python reference output matrices."
    )

    parser.add_argument("--H", type=int, default=64, help="Image height used for this comparison.")
    parser.add_argument("--W", type=int, default=64, help="Image width used for this comparison.")
    parser.add_argument("--Cin", type=int, default=3, help="Number of input channels.")
    parser.add_argument("--Cout", type=int, default=8, help="Number of output filters.")
    parser.add_argument("--K", type=int, default=3, help="Kernel size.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for PASS/FAIL.")

    parser.add_argument("--seq-dir", type=Path, default=Path("../sequential_implementation"))
    parser.add_argument("--openmp-dir", type=Path, default=Path("../openmp"))
    parser.add_argument("--cuda-naive-dir", type=Path, default=Path("../cuda_naive"))
    parser.add_argument("--cuda-shared-dir", type=Path, default=Path("../cuda_shared"))
    parser.add_argument("--python-ref-dir", type=Path, default=Path("../python_ref"))

    parser.add_argument("--summary-csv", type=Path, default=Path("comparison_summary.csv"))
    parser.add_argument("--input-check-csv", type=Path, default=Path("input_kernel_consistency.csv"))

    args = parser.parse_args()

    method_dirs = {
        "sequential": args.seq_dir,
        "openmp": args.openmp_dir,
        "cuda_naive": args.cuda_naive_dir,
        "cuda_shared": args.cuda_shared_dir,
        "python_reference": args.python_ref_dir,
    }

    method_prefixes = {
        "sequential": "sequential",
        "openmp": "openmp",
        "cuda_naive": "cuda_naive",
        "cuda_shared": "cuda_shared",
    }

    print("ME759 CNN convolution output comparison")
    print("--------------------------------------")
    print(f"Case: H={args.H}, W={args.W}, Cin={args.Cin}, Cout={args.Cout}, K={args.K}")
    print(f"Tolerance: {args.tol:.1e}")
    print()

    print("Checking input.csv and kernel.csv consistency across folders...")
    input_ok = check_input_kernel_files(method_dirs, "python_reference", args.input_check_csv)
    print(f"Input/kernel consistency: {'PASS' if input_ok else 'FAIL'}")
    print(f"Wrote: {args.input_check_csv}")
    print()

    summary_rows = []
    overall_pass = True

    for co in range(args.Cout):
        ref_file = args.python_ref_dir / f"python_reference_filter_{co}.csv"
        try:
            ref_matrix = read_matrix_csv(ref_file)
        except Exception as exc:
            print(f"ERROR: could not read reference file for filter {co}: {exc}")
            overall_pass = False
            continue

        for method, prefix in method_prefixes.items():
            test_file = method_dirs[method] / f"{prefix}_filter_{co}.csv"

            try:
                test_matrix = read_matrix_csv(test_file)
                max_abs_diff, mean_abs_diff, rmse, compared, expected = compare_matrices(ref_matrix, test_matrix)
                mismatch_count = count_exceeding(ref_matrix, test_matrix, args.tol)
                status = "PASS" if max_abs_diff <= args.tol else "FAIL"

                if status == "FAIL":
                    overall_pass = False

                summary_rows.append([
                    method,
                    co,
                    str(test_file),
                    str(ref_file),
                    len(ref_matrix),
                    len(ref_matrix[0]),
                    compared,
                    f"{max_abs_diff:.10e}",
                    f"{mean_abs_diff:.10e}",
                    f"{rmse:.10e}",
                    mismatch_count,
                    args.tol,
                    status,
                ])

                print(
                    f"Filter {co:02d} | {method:11s} | {status:4s} | "
                    f"max_abs_diff={max_abs_diff:.10e}, mean_abs_diff={mean_abs_diff:.10e}, "
                    f"rmse={rmse:.10e}, mismatches>{args.tol:.1e}: {mismatch_count}"
                )

            except Exception as exc:
                overall_pass = False
                summary_rows.append([
                    method,
                    co,
                    str(test_file),
                    str(ref_file),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    args.tol,
                    f"ERROR: {exc}",
                ])
                print(f"Filter {co:02d} | {method:11s} | ERROR | {exc}")

    with args.summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "filter",
            "test_file",
            "reference_file",
            "rows",
            "cols",
            "num_values_compared",
            "max_abs_diff",
            "mean_abs_diff",
            "rmse",
            "mismatch_count_above_tol",
            "tolerance",
            "status",
        ])
        writer.writerows(summary_rows)

    print()
    print(f"Wrote: {args.summary_csv}")
    print(f"FINAL RESULT: {'PASS' if overall_pass and input_ok else 'FAIL'}")

    return 0 if overall_pass and input_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
