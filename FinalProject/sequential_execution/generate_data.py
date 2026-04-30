#!/usr/bin/env python3
"""
Generate shared image/input data and convolution weight/kernel data.

This script creates:
  input.csv   containing H*W*Cin values in input[ci][h][w] order
  kernel.csv  containing Cout*Cin*K*K values in kernel[co][ci][kh][kw] order

All tasks should use the same generated input.csv and kernel.csv.
"""

import argparse
import random
from pathlib import Path


def write_csv_flat(path: Path, values):
    with path.open("w") as f:
        for i, v in enumerate(values):
            if i > 0:
                f.write(",")
            f.write(f"{v:.8f}")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=64)
    parser.add_argument("--Cin", type=int, default=3)
    parser.add_argument("--Cout", type=int, default=8)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--seed", type=int, default=759)
    parser.add_argument("--input", type=str, default="input.csv")
    parser.add_argument("--kernel", type=str, default="kernel.csv")
    args = parser.parse_args()

    if args.H <= 0 or args.W <= 0 or args.Cin <= 0 or args.Cout <= 0 or args.K <= 0:
        raise ValueError("H, W, Cin, Cout, and K must be positive.")

    random.seed(args.seed)

    input_count = args.H * args.W * args.Cin
    kernel_count = args.Cout * args.Cin * args.K * args.K

    # Image/input values: deterministic pseudo-random values in [0, 1).
    input_values = [random.random() for _ in range(input_count)]

    # Kernel/weight values: small deterministic pseudo-random weights around zero.
    kernel_values = [(random.random() - 0.5) * 0.2 for _ in range(kernel_count)]

    write_csv_flat(Path(args.input), input_values)
    write_csv_flat(Path(args.kernel), kernel_values)

    print("Generated shared data files")
    print(f"  input:  {args.input}  values={input_count}")
    print(f"  kernel: {args.kernel}  values={kernel_count}")
    print(f"  H={args.H}, W={args.W}, Cin={args.Cin}, Cout={args.Cout}, K={args.K}, seed={args.seed}")


if __name__ == "__main__":
    main()
