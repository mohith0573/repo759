#!/usr/bin/env python3
"""
Python reference implementation for forward/right-bottom 2D convolution.

Convolution rule:
    output[co][h][w] = sum_{ci,kh,kw} input[ci][h+kh][w+kw] * kernel[co][ci][kh][kw]

Boundary rule:
    If h+kh >= H or w+kw >= W, that contribution is treated as zero.

Input layout in input.csv:
    input[ci][h][w] flattened as (ci * H + h) * W + w

Kernel layout in kernel.csv:
    kernel[co][ci][kh][kw] flattened as ((co * Cin + ci) * K + kh) * K + kw

Output layout:
    output[co][h][w] flattened as (co * H + h) * W + w
"""

import argparse
import math
import sys
import time
from pathlib import Path


def read_flat_csv(path: str, expected_count: int, name: str):
    """Read a comma/newline separated flat CSV file into a list of floats."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    text = p.read_text().strip()
    if not text:
        raise ValueError(f"{name} file is empty: {path}")

    # Accept either one-line CSV or multi-line CSV.
    tokens = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            tokens.append(part)

    if len(tokens) != expected_count:
        raise ValueError(
            f"{name} file {path} has {len(tokens)} values, but expected {expected_count}.\n"
            f"Check H, W, Cin, Cout, and K arguments."
        )

    try:
        return [float(x) for x in tokens]
    except ValueError as exc:
        raise ValueError(f"{name} file contains a non-numeric value: {path}") from exc


def write_output_matrices(output, H: int, W: int, Cout: int, prefix: str):
    """Write one HxW output matrix per output filter."""
    for co in range(Cout):
        out_name = f"{prefix}_filter_{co}.csv"
        with open(out_name, "w", encoding="utf-8") as f:
            base = co * H * W
            for h in range(H):
                row_start = base + h * W
                row = output[row_start: row_start + W]
                # 8 decimals is enough for validation while keeping files readable.
                f.write(",".join(f"{v:.8f}" for v in row))
                f.write("\n")


def conv2d_forward_reference(input_data, kernel, H: int, W: int, Cin: int, Cout: int, K: int):
    """
    Locality-aware pure Python reference.

    For each PE/output pixel, keep all Cout accumulators locally.
    For each loaded input pixel, reuse that value across all output filters.
    """
    output = [0.0] * (Cout * H * W)
    HW = H * W
    KK = K * K

    for h in range(H):
        for w in range(W):
            acc = [0.0] * Cout

            for ci in range(Cin):
                input_channel_base = ci * HW

                for kh in range(K):
                    ih = h + kh
                    if ih >= H:
                        continue

                    input_row_base = input_channel_base + ih * W

                    for kw in range(K):
                        iw = w + kw
                        if iw >= W:
                            continue

                        val = input_data[input_row_base + iw]

                        # Reuse this one input value across all output filters.
                        for co in range(Cout):
                            kidx = ((co * Cin + ci) * K + kh) * K + kw
                            acc[co] += val * kernel[kidx]

            out_pixel_offset = h * W + w
            for co in range(Cout):
                output[co * HW + out_pixel_offset] = acc[co]

    return output


def checksum_output(output):
    """Numerical checksum for quick consistency checking."""
    return math.fsum(output)


def parse_args():
    parser = argparse.ArgumentParser(description="Python reference forward convolution")
    parser.add_argument("--H", type=int, required=True, help="Image height")
    parser.add_argument("--W", type=int, required=True, help="Image width")
    parser.add_argument("--Cin", type=int, required=True, help="Number of input channels")
    parser.add_argument("--Cout", type=int, required=True, help="Number of output filters")
    parser.add_argument("--K", type=int, required=True, help="Kernel size")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs for timing")
    parser.add_argument("--input", default="input.csv", help="Input CSV file")
    parser.add_argument("--kernel", default="kernel.csv", help="Kernel CSV file")
    parser.add_argument("--write-matrices", type=int, default=1, help="1 writes output matrix CSV files, 0 disables")
    parser.add_argument("--prefix", default="python_reference", help="Prefix for output matrix CSV files")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.H <= 0 or args.W <= 0 or args.Cin <= 0 or args.Cout <= 0 or args.K <= 0:
        print("ERROR: H, W, Cin, Cout, and K must all be positive.", file=sys.stderr)
        sys.exit(1)

    if args.repeats <= 0:
        print("ERROR: repeats must be positive.", file=sys.stderr)
        sys.exit(1)

    input_count = args.H * args.W * args.Cin
    kernel_count = args.Cout * args.Cin * args.K * args.K

    input_data = read_flat_csv(args.input, input_count, "input")
    kernel = read_flat_csv(args.kernel, kernel_count, "kernel")

    output = None
    start = time.perf_counter()
    for _ in range(args.repeats):
        output = conv2d_forward_reference(input_data, kernel, args.H, args.W, args.Cin, args.Cout, args.K)
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000.0
    avg_time_ms = total_time_ms / args.repeats
    checksum = checksum_output(output)

    if args.write_matrices:
        write_output_matrices(output, args.H, args.W, args.Cout, args.prefix)

    print("method,H,W,Cin,Cout,K,repeats,time_ms,total_time_ms,checksum,input_file,kernel_file")
    print(
        f"python_reference,{args.H},{args.W},{args.Cin},{args.Cout},{args.K},"
        f"{args.repeats},{avg_time_ms:.6f},{total_time_ms:.6f},{checksum:.6f},"
        f"{args.input},{args.kernel}"
    )


if __name__ == "__main__":
    main()
