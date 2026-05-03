#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

METHODS = ["sequential", "openmp", "cuda_naive", "cuda_shared", "python_reference"]
LABELS = {"sequential":"Sequential", "openmp":"OpenMP", "cuda_naive":"CUDA Naive", "cuda_shared":"CUDA Shared", "python_reference":"Python Reference"}
PREFIXES = {
    "sequential": ["sequential_results"],
    "openmp": ["openmp_results"],
    "cuda_naive": ["cuda_naive_results"],
    "cuda_shared": ["cuda_shared_results"],
    "python_reference": ["python_reference_results", "python_refernce_results"],
}


def read_row(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[-1]


def find_file(exe_dir: Path, method: str, size: int) -> Path | None:
    for pfx in PREFIXES[method]:
        p = exe_dir / f"{pfx}_{size}.csv"
        if p.exists():
            return p
    return None


def to_f(x, default=math.nan):
    if x is None or x == "": return default
    return float(x)

def to_i(x, default=0):
    if x is None or x == "": return default
    return int(float(x))


def plot_time(method: str, row: dict) -> float:
    return to_f(row.get("kernel_time_ms")) if method.startswith("cuda") else to_f(row.get("time_ms"))


def flops(H,W,Cin,Cout,K):
    return float(H*W*Cin*Cout*K*K*2)


def ceil_div(a,b): return (a+b-1)//b


def bytes_naive(H,W,Cin,Cout,K):
    out = H*W*Cout
    return float(out*Cin*K*K*4 + out*Cin*K*K*4 + out*4)  # input + weights + output


def bytes_shared(H,W,Cin,Cout,K,tile=16):
    nbx, nby = ceil_div(W,tile), ceil_div(H,tile)
    input_tile = nbx*nby*Cin*(tile+K-1)*(tile+K-1)*4
    weight = H*W*Cout*Cin*K*K*4
    output = H*W*Cout*4
    return float(input_tile + weight + output)


def load_all(exe_dir: Path, sizes: list[int]) -> list[dict]:
    rows=[]
    for size in sizes:
        for method in METHODS:
            p = find_file(exe_dir, method, size)
            if p is None:
                print(f"WARNING: missing {method} results for size {size}")
                continue
            r = read_row(p)
            row = {
                "size": size, "method": method,
                "H": to_i(r.get("H"), size), "W": to_i(r.get("W"), size),
                "Cin": to_i(r.get("Cin"), 16), "Cout": to_i(r.get("Cout"), 8), "K": to_i(r.get("K"), 3),
                "threads": r.get("threads", ""), "threads_per_block": r.get("threads_per_block", ""),
                "repeats": r.get("repeats", ""), "time_ms": r.get("time_ms", ""),
                "kernel_time_ms": r.get("kernel_time_ms", ""), "total_time_ms": r.get("total_time_ms", ""),
                "plot_time_ms": f"{plot_time(method,r):.10f}", "checksum": r.get("checksum", ""),
                "input_file": r.get("input_file", ""), "kernel_file": r.get("kernel_file", ""),
                "gpu_name": r.get("gpu_name", ""), "source_csv": str(p)
            }
            rows.append(row)
    return rows


def write_csv(name, rows, fields):
    with open(name, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def grouped(rows):
    g=defaultdict(dict)
    for r in rows: g[int(r["size"])][r["method"]]=r
    return g


def save_checksum(rows):
    g=grouped(rows)
    with open("checksum_summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["size","method","checksum","python_reference_checksum","abs_diff","status"])
        for size in sorted(g):
            ref = to_f(g[size].get("python_reference",{}).get("checksum"))
            for m,r in g[size].items():
                c=to_f(r.get("checksum"))
                diff=abs(c-ref) if not (math.isnan(c) or math.isnan(ref)) else math.nan
                status="PASS" if not math.isnan(diff) and diff<=1e-4 else "CHECK"
                w.writerow([size,m,r.get("checksum",""),"" if math.isnan(ref) else f"{ref:.10f}","" if math.isnan(diff) else f"{diff:.10e}",status])


def line_plot(rows, methods, ylabel, title, outfile, use_total=False):
    g=grouped(rows); sizes=sorted(g)
    plt.figure(figsize=(9,5))
    for m in methods:
        x=[]; y=[]
        for s in sizes:
            r=g[s].get(m)
            if not r: continue
            val = to_f(r.get("total_time_ms")) if use_total else to_f(r.get("plot_time_ms"))
            if math.isnan(val): continue
            x.append(s); y.append(val)
        if y: plt.plot(x,y,marker="o",label=LABELS[m])
    plt.xlabel("Image Size (N for N×N)"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5); plt.legend(); plt.tight_layout(); plt.savefig(outfile,dpi=300); plt.close()


def speedup_plot(rows):
    g=grouped(rows); sizes=sorted(g)
    plt.figure(figsize=(9,5))
    for m in ["sequential","openmp","cuda_naive","cuda_shared"]:
        x=[]; y=[]
        for s in sizes:
            seq=g[s].get("sequential"); r=g[s].get(m)
            if not seq or not r: continue
            st, mt = to_f(seq["plot_time_ms"]), to_f(r["plot_time_ms"])
            if mt>0: x.append(s); y.append(st/mt)
        if y: plt.plot(x,y,marker="o",label=LABELS[m])
    plt.xlabel("Image Size (N for N×N)"); plt.ylabel("Speedup vs Sequential"); plt.title("Speedup vs Sequential")
    plt.grid(True, linestyle="--", linewidth=0.5); plt.legend(); plt.tight_layout(); plt.savefig("speedup_vs_image_size.png",dpi=300); plt.close()


def roofline(rows):
    out=[]
    for r in rows:
        m=r["method"]
        if m not in ("cuda_naive","cuda_shared"): continue
        H,W,Cin,Cout,K = int(r["H"]),int(r["W"]),int(r["Cin"]),int(r["Cout"]),int(r["K"])
        kt=to_f(r.get("kernel_time_ms"))
        if kt<=0 or math.isnan(kt): continue
        F=flops(H,W,Cin,Cout,K); B=bytes_naive(H,W,Cin,Cout,K) if m=="cuda_naive" else bytes_shared(H,W,Cin,Cout,K)
        ai=F/B; gflops=F/(kt/1000.0)/1e9
        out.append({"size":r["size"],"method":m,"H":H,"W":W,"Cin":Cin,"Cout":Cout,"K":K,"kernel_time_ms":f"{kt:.10f}","FLOPs":f"{F:.6e}","estimated_bytes_moved":f"{B:.6e}","arithmetic_intensity_FLOPs_per_byte":f"{ai:.6e}","achieved_GFLOPs_per_s":f"{gflops:.6e}"})
    if out:
        write_csv("roofline_cuda.csv", out, list(out[0].keys()))
    plt.figure(figsize=(8,5))
    for m in ["cuda_naive","cuda_shared"]:
        xs=[]; ys=[]; labs=[]
        for r in out:
            if r["method"]==m:
                xs.append(to_f(r["arithmetic_intensity_FLOPs_per_byte"])); ys.append(to_f(r["achieved_GFLOPs_per_s"])); labs.append(str(r["size"]))
        if xs:
            plt.scatter(xs,ys,label=LABELS[m])
            for x,y,lab in zip(xs,ys,labs): plt.annotate(lab,(x,y),textcoords="offset points",xytext=(4,4),fontsize=8)
    plt.xlabel("Arithmetic Intensity (FLOPs/byte)"); plt.ylabel("Achieved Performance (GFLOP/s)"); plt.title("Roofline-Style CUDA Analysis")
    plt.xscale("log"); plt.yscale("log"); plt.grid(True, linestyle="--", linewidth=0.5); plt.legend(); plt.tight_layout(); plt.savefig("roofline_cuda.png",dpi=300); plt.close()
    # GFLOPs vs size
    gg=defaultdict(dict)
    for r in out: gg[int(r["size"])][r["method"]]=to_f(r["achieved_GFLOPs_per_s"])
    plt.figure(figsize=(8,5))
    for m in ["cuda_naive","cuda_shared"]:
        x=[]; y=[]
        for s in sorted(gg):
            if m in gg[s]: x.append(s); y.append(gg[s][m])
        if y: plt.plot(x,y,marker="o",label=LABELS[m])
    plt.xlabel("Image Size (N for N×N)"); plt.ylabel("Achieved GFLOP/s"); plt.title("CUDA Achieved GFLOP/s vs Image Size")
    plt.grid(True, linestyle="--", linewidth=0.5); plt.legend(); plt.tight_layout(); plt.savefig("gflops_vs_image_size.png",dpi=300); plt.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--exe-dir",type=Path,default=Path("../exe_results")); ap.add_argument("--sizes",nargs="+",type=int,default=[64,128,256,512]); args=ap.parse_args()
    rows=load_all(args.exe_dir,args.sizes)
    if not rows: raise SystemExit("No rows found")
    fields=["size","method","H","W","Cin","Cout","K","threads","threads_per_block","repeats","time_ms","kernel_time_ms","total_time_ms","plot_time_ms","checksum","input_file","kernel_file","gpu_name","source_csv"]
    write_csv("benchmark_summary.csv",rows,fields); save_checksum(rows)
    line_plot(rows,["sequential","openmp","cuda_naive","cuda_shared"],"Execution Time (ms)","Execution Time vs Image Size","execution_time_vs_image_size.png")
    speedup_plot(rows)
    line_plot(rows,["cuda_naive","cuda_shared"],"CUDA Kernel Time (ms)","CUDA Naive vs Shared Memory","cuda_naive_vs_shared_kernel_time.png")
    line_plot(rows,["cuda_naive","cuda_shared"],"CUDA Total Time (ms)","CUDA Total Time vs Image Size","cuda_total_time_vs_image_size.png",use_total=True)
    line_plot(rows,["sequential","python_reference"],"Time (ms)","Python Reference vs Sequential C++","python_reference_time.png")
    roofline(rows)
    print("Generated plots and CSVs in current folder.")

if __name__=="__main__": main()
