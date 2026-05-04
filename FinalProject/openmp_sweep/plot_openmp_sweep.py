#!/usr/bin/env python3
"""
Generate OpenMP strong-scaling plots from exe_results/openmp_sweep_all.csv
or exe_results/openmp_sweep_<size>.csv files.

Run from:
    FinalProject/openmp_sweep/

Manual:
    python3 plot_openmp_sweep.py

SLURM:
    sbatch plot_openmp_sweep_job.sh
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_sweep_rows(exe_results_dir: Path):
    rows = []

    combined = exe_results_dir / "openmp_sweep_all.csv"
    if combined.exists():
        files = [combined]
    else:
        files = sorted(exe_results_dir.glob("openmp_sweep_*.csv"))

    for file in files:
        if file.name.startswith("openmp_sweep_all"):
            pass
        data = read_csv(file)
        for r in data:
            if not r:
                continue
            try:
                rows.append({
                    "H": int(r["H"]),
                    "W": int(r["W"]),
                    "size": int(r["H"]),
                    "Cin": int(r["Cin"]),
                    "Cout": int(r["Cout"]),
                    "K": int(r["K"]),
                    "threads": int(r["threads"]),
                    "time_ms": float(r["time_ms"]),
                    "total_time_ms": float(r["total_time_ms"]),
                    "checksum": float(r["checksum"]),
                    "source_file": str(file),
                })
            except Exception as exc:
                print(f"Skipping row in {file}: {exc}")

    return rows


def write_scaling_summary(rows, out_path: Path):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["size"]].append(r)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "size", "threads", "time_ms",
            "speedup_vs_1_thread", "parallel_efficiency",
            "checksum"
        ])

        for size, srows in sorted(grouped.items()):
            srows = sorted(srows, key=lambda x: x["threads"])
            one_thread = next((x for x in srows if x["threads"] == 1), srows[0])
            base_time = one_thread["time_ms"]

            for r in srows:
                speedup = base_time / r["time_ms"]
                efficiency = speedup / r["threads"]
                writer.writerow([
                    size,
                    r["threads"],
                    f"{r['time_ms']:.10f}",
                    f"{speedup:.10f}",
                    f"{efficiency:.10f}",
                    f"{r['checksum']:.10f}",
                ])


def plot_time(rows, out_dir: Path):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["size"]].append(r)

    plt.figure(figsize=(8.5, 5.2))

    for size, srows in sorted(grouped.items()):
        srows = sorted(srows, key=lambda x: x["threads"])
        xs = [r["threads"] for r in srows]
        ys = [r["time_ms"] for r in srows]
        plt.plot(xs, ys, marker="o", label=f"{size}x{size}")

    plt.xlabel("OpenMP threads")
    plt.ylabel("Execution time (ms)")
    plt.title("OpenMP Execution Time vs Thread Count")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "openmp_time_vs_threads.pdf")
    plt.close()


def plot_speedup(rows, out_dir: Path):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["size"]].append(r)

    plt.figure(figsize=(8.5, 5.2))

    for size, srows in sorted(grouped.items()):
        srows = sorted(srows, key=lambda x: x["threads"])
        one_thread = next((x for x in srows if x["threads"] == 1), srows[0])
        base_time = one_thread["time_ms"]

        xs = [r["threads"] for r in srows]
        ys = [base_time / r["time_ms"] for r in srows]
        plt.plot(xs, ys, marker="o", label=f"{size}x{size}")

    # Ideal line based on maximum available thread count.
    all_threads = sorted({r["threads"] for r in rows})
    if all_threads:
        plt.plot(all_threads, all_threads, linestyle="--", label="Ideal")

    plt.xlabel("OpenMP threads")
    plt.ylabel("Speedup vs 1 thread")
    plt.title("OpenMP Strong Scaling")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "openmp_strong_scaling.pdf")
    plt.close()


def plot_efficiency(rows, out_dir: Path):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["size"]].append(r)

    plt.figure(figsize=(8.5, 5.2))

    for size, srows in sorted(grouped.items()):
        srows = sorted(srows, key=lambda x: x["threads"])
        one_thread = next((x for x in srows if x["threads"] == 1), srows[0])
        base_time = one_thread["time_ms"]

        xs = [r["threads"] for r in srows]
        ys = [(base_time / r["time_ms"]) / r["threads"] for r in srows]
        plt.plot(xs, ys, marker="o", label=f"{size}x{size}")

    plt.xlabel("OpenMP threads")
    plt.ylabel("Parallel efficiency")
    plt.title("OpenMP Parallel Efficiency")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "openmp_efficiency.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-results-dir", type=Path, default=Path("../exe_results"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    rows = load_sweep_rows(args.exe_results_dir)

    if not rows:
        raise SystemExit(f"No OpenMP sweep rows found in {args.exe_results_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_scaling_summary(rows, args.out_dir / "openmp_scaling_summary.csv")
    plot_time(rows, args.out_dir)
    plot_speedup(rows, args.out_dir)
    plot_efficiency(rows, args.out_dir)

    print("Generated:")
    print(args.out_dir / "openmp_scaling_summary.csv")
    print(args.out_dir / "openmp_time_vs_threads.pdf")
    print(args.out_dir / "openmp_strong_scaling.pdf")
    print(args.out_dir / "openmp_efficiency.pdf")


if __name__ == "__main__":
    main()
