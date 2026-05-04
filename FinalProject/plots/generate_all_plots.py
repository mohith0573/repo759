#!/usr/bin/env python3
"""
Generate ME759 final project benchmark plots and a PDF benchmark table.

This version intentionally excludes OpenMP thread-scaling plots and OpenMP sweep scripts.

Expected project layout:

Project/
├── exe_results/
│   ├── sequential_results_64.csv
│   ├── openmp_results_64.csv
│   ├── cuda_naive_results_64.csv
│   ├── cuda_shared_results_64.csv
│   ├── python_reference_results_64.csv or python_refernce_results_64.csv
│   ├── sequential_results_128.csv
│   ├── openmp_results_128.csv
│   ├── cuda_naive_results_128.csv
│   ├── cuda_shared_results_128.csv
│   ├── ...
└── plots/
    ├── generate_all_plots.py
    ├── plots_job.sh
    └── ...

Run from inside plots/:
    python3 generate_all_plots.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

METHOD_DISPLAY = {
    "sequential": "Sequential",
    "openmp": "OpenMP",
    "cuda_naive": "CUDA Naive",
    "cuda_shared": "CUDA Shared",
    "python_reference": "Python Reference",
    "python_ref": "Python Reference",
}

METHOD_ORDER = ["sequential", "openmp", "cuda_naive", "cuda_shared", "python_reference"]


def method_key(method: str) -> str:
    m = method.strip().lower()
    if m in {"python_refernce", "python_reference", "python_ref"}:
        return "python_reference"
    return m


def display_name(method: str) -> str:
    return METHOD_DISPLAY.get(method_key(method), method)


def read_csv_rows(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def infer_size_from_filename(path: Path):
    m = re.search(r"_(\d+)\.csv$", path.name)
    if m:
        return int(m.group(1))
    return None


def get_time_ms(row):
    """CPU/Python rows use time_ms. CUDA rows use kernel_time_ms."""
    if "kernel_time_ms" in row and row.get("kernel_time_ms", "") not in {"", None}:
        return float(row["kernel_time_ms"])
    return float(row["time_ms"])


def get_total_time_ms(row):
    if "total_time_ms" in row and row.get("total_time_ms", "") not in {"", None}:
        return float(row["total_time_ms"])
    return get_time_ms(row)


def parse_benchmark_results(exe_results_dir: Path):
    rows = []
    if not exe_results_dir.exists():
        raise FileNotFoundError(f"Missing directory: {exe_results_dir}")

    for path in sorted(exe_results_dir.glob("*.csv")):
        if path.name.startswith("combined_") or path.name.startswith("benchmark_"):
            continue
        if "sweep" in path.name.lower():
            continue

        try:
            file_rows = read_csv_rows(path)
        except Exception as exc:
            print(f"Skipping {path}: could not read CSV: {exc}")
            continue

        for r in file_rows:
            if not r or "method" not in r:
                continue

            method = method_key(r["method"])
            size = None

            if "H" in r and r["H"]:
                try:
                    size = int(r["H"])
                except ValueError:
                    pass

            if size is None:
                size = infer_size_from_filename(path)

            if size is None:
                print(f"Skipping row from {path}: could not infer size.")
                continue

            try:
                H = int(r.get("H", size))
                W = int(r.get("W", size))
                Cin = int(r.get("Cin", 0))
                Cout = int(r.get("Cout", 0))
                K = int(r.get("K", 0))
                time_ms = get_time_ms(r)
                total_time_ms = get_total_time_ms(r)
                checksum = float(r.get("checksum", "nan"))
            except Exception as exc:
                print(f"Skipping row from {path}: parse error: {exc}")
                continue

            rows.append({
                "method": method,
                "method_display": display_name(method),
                "size": size,
                "H": H,
                "W": W,
                "Cin": Cin,
                "Cout": Cout,
                "K": K,
                "time_ms": time_ms,
                "total_time_ms": total_time_ms,
                "checksum": checksum,
                "source_file": str(path),
            })

    return rows


def save_combined_csv(rows, out_path: Path):
    columns = [
        "method", "size", "H", "W", "Cin", "Cout", "K",
        "time_ms", "total_time_ms", "checksum", "source_file"
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def rows_by_size_method(rows):
    d = {}
    for r in rows:
        d[(r["size"], r["method"])] = r
    return d


def plot_execution_time(rows, out_dir: Path):
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    sizes = sorted({r["size"] for r in rows})

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        xs, ys = [], []
        for size in sizes:
            match = [r for r in rows if r["size"] == size and r["method"] == method]
            if match:
                xs.append(size)
                ys.append(match[0]["time_ms"])
        if xs:
            plt.plot(xs, ys, marker="o", label=display_name(method))

    plt.xlabel("Image size N for N x N image")
    plt.ylabel("Execution time (ms)")
    plt.title("Execution Time vs Image Size")
    plt.yscale("log")
    plt.grid(True, which="both", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "execution_time_by_image_size.pdf")
    plt.close()


def plot_total_time(rows, out_dir: Path):
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    sizes = sorted({r["size"] for r in rows})

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        xs, ys = [], []
        for size in sizes:
            match = [r for r in rows if r["size"] == size and r["method"] == method]
            if match:
                xs.append(size)
                ys.append(match[0]["total_time_ms"])
        if xs:
            plt.plot(xs, ys, marker="o", label=display_name(method))

    plt.xlabel("Image size N for N x N image")
    plt.ylabel("Total time (ms)")
    plt.title("End-to-End Total Time vs Image Size")
    plt.yscale("log")
    plt.grid(True, which="both", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "total_time_by_image_size.pdf")
    plt.close()


def plot_speedup(rows, out_dir: Path):
    d = rows_by_size_method(rows)
    sizes = sorted({r["size"] for r in rows})
    methods = ["openmp", "cuda_naive", "cuda_shared"]

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        xs, ys = [], []
        for size in sizes:
            seq = d.get((size, "sequential"))
            cur = d.get((size, method))
            if seq and cur and cur["time_ms"] > 0:
                xs.append(size)
                ys.append(seq["time_ms"] / cur["time_ms"])
        if xs:
            plt.plot(xs, ys, marker="o", label=display_name(method))

    plt.xlabel("Image size N for N x N image")
    plt.ylabel("Speedup over sequential")
    plt.title("Speedup vs Sequential Baseline")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_by_image_size.pdf")
    plt.close()


def plot_cuda_comparison(rows, out_dir: Path):
    sizes = sorted({r["size"] for r in rows})
    d = rows_by_size_method(rows)

    naive, shared, valid_sizes = [], [], []
    for size in sizes:
        n = d.get((size, "cuda_naive"))
        s = d.get((size, "cuda_shared"))
        if n and s:
            valid_sizes.append(size)
            naive.append(n["time_ms"])
            shared.append(s["time_ms"])

    if not valid_sizes:
        return

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(valid_sizes, naive, marker="o", label="CUDA Naive")
    plt.plot(valid_sizes, shared, marker="o", label="CUDA Shared")
    plt.xlabel("Image size N for N x N image")
    plt.ylabel("CUDA kernel time (ms)")
    plt.title("CUDA Naive vs CUDA Shared-Memory Tiled")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cuda_naive_vs_shared_by_image_size.pdf")
    plt.close()


def flops(row):
    return 2.0 * row["H"] * row["W"] * row["Cin"] * row["Cout"] * row["K"] * row["K"]


def estimated_bytes(row):
    H, W, Cin, Cout, K = row["H"], row["W"], row["Cin"], row["Cout"], row["K"]
    method = row["method"]
    bytes_per_float = 4.0
    tile = 16

    if method == "cuda_shared":
        blocks_x = math.ceil(W / tile)
        blocks_y = math.ceil(H / tile)
        input_reads = blocks_x * blocks_y * (tile + K - 1) * (tile + K - 1) * Cin
    else:
        input_reads = H * W * Cin * K * K

    kernel_reads = H * W * Cout * Cin * K * K
    output_writes = H * W * Cout

    return (input_reads + kernel_reads + output_writes) * bytes_per_float


def plot_roofline_style(rows, out_dir: Path):
    cuda_rows = [r for r in rows if r["method"] in {"cuda_naive", "cuda_shared"}]
    if not cuda_rows:
        return

    plt.figure(figsize=(8.0, 5.5))

    for method in ["cuda_naive", "cuda_shared"]:
        xs, ys, labels = [], [], []
        for r in cuda_rows:
            if r["method"] != method:
                continue
            f = flops(r)
            b = estimated_bytes(r)
            if r["time_ms"] <= 0 or b <= 0:
                continue
            gflops = f / (r["time_ms"] / 1000.0) / 1e9
            ai = f / b
            xs.append(ai)
            ys.append(gflops)
            labels.append(str(r["size"]))

        if xs:
            plt.scatter(xs, ys, label=display_name(method))
            for x, y, label in zip(xs, ys, labels):
                plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.xlabel("Estimated arithmetic intensity (FLOPs/byte)")
    plt.ylabel("Achieved performance (GFLOP/s)")
    plt.title("Roofline-Style CUDA Kernel Positioning")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roofline_style_cuda.pdf")
    plt.close()

    table_path = out_dir / "roofline_cuda_table.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "size", "H", "W", "Cin", "Cout", "K",
            "kernel_time_ms", "FLOPs", "estimated_bytes", "arithmetic_intensity", "GFLOP_per_s"
        ])
        for r in cuda_rows:
            fops = flops(r)
            b = estimated_bytes(r)
            ai = fops / b if b else 0
            gflops = fops / (r["time_ms"] / 1000.0) / 1e9 if r["time_ms"] > 0 else 0
            writer.writerow([
                r["method"], r["size"], r["H"], r["W"], r["Cin"], r["Cout"], r["K"],
                r["time_ms"], f"{fops:.6e}", f"{b:.6e}", f"{ai:.6e}", f"{gflops:.6e}"
            ])


def plot_benchmark_table(rows, out_dir: Path):
    sizes = sorted({r["size"] for r in rows})
    d = rows_by_size_method(rows)

    methods = ["sequential", "openmp", "cuda_naive", "cuda_shared", "python_reference"]
    available_methods = [m for m in methods if any(r["method"] == m for r in rows)]
    columns = ["Image size"] + [display_name(m) for m in available_methods]

    table_data = []
    for size in sizes:
        row = [f"{size}x{size}"]
        for method in available_methods:
            r = d.get((size, method))
            row.append(f"{r['time_ms']:.6g} ms" if r else "-")
        table_data.append(row)

    fig_height = max(3.0, 0.45 * len(table_data) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    table = ax.table(cellText=table_data, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    ax.set_title("Benchmark Timing Summary\nCPU/Python use time_ms; CUDA uses kernel_time_ms", pad=16)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_summary_table.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-results-dir", type=Path, default=Path("../exe_results"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_benchmark_results(args.exe_results_dir)
    if not rows:
        raise SystemExit(f"No benchmark rows found in {args.exe_results_dir}")

    combined_csv = out_dir / "combined_benchmark_results.csv"
    save_combined_csv(rows, combined_csv)

    plot_execution_time(rows, out_dir)
    plot_total_time(rows, out_dir)
    plot_speedup(rows, out_dir)
    plot_cuda_comparison(rows, out_dir)
    plot_roofline_style(rows, out_dir)
    plot_benchmark_table(rows, out_dir)

    print("Generated plots/tables:")
    for name in [
        "combined_benchmark_results.csv",
        "execution_time_by_image_size.pdf",
        "total_time_by_image_size.pdf",
        "speedup_by_image_size.pdf",
        "cuda_naive_vs_shared_by_image_size.pdf",
        "roofline_style_cuda.pdf",
        "roofline_cuda_table.csv",
        "benchmark_summary_table.pdf",
    ]:
        p = out_dir / name
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
