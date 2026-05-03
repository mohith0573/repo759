# How to Generate Plots

Put this folder as:

```text
plots/
```

Your project should look like:

```text
inputs/
sequential_execution/
openmp/
cuda_naive/
cuda_shared/
python_ref/
results_comparison/
exe_results/
plots/
```

## Run manually

From inside `plots/`:

```bash
python3 plot_from_exe_results.py --exe-dir ../exe_results --sizes 64 128 256 512
```

## Run with SLURM

```bash
sbatch plot_job.sh
```

## Generated files

```text
benchmark_summary.csv
checksum_summary.csv
execution_time_vs_image_size.png
speedup_vs_image_size.png
cuda_naive_vs_shared_kernel_time.png
cuda_total_time_vs_image_size.png
python_reference_time.png
roofline_cuda.csv
roofline_cuda.png
gflops_vs_image_size.png
```

## What each plot means

### execution_time_vs_image_size.png

Shows runtime scaling as image size increases.

### speedup_vs_image_size.png

Uses:

```text
speedup = sequential_time / method_time
```

### cuda_naive_vs_shared_kernel_time.png

Compares CUDA naive and CUDA shared-memory kernel time.

### roofline_cuda.png

Uses:

```text
FLOPs = H × W × Cout × Cin × K × K × 2
```

and estimates arithmetic intensity:

```text
FLOPs / estimated bytes moved
```

### gflops_vs_image_size.png

Shows achieved CUDA computational throughput.
