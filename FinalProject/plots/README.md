# Plots From `exe_results`

This folder generates plots from your saved execution-result CSV files in `exe_results/`.

Your benchmark cases are:

```text
64×64,  Cin=16, Cout=8, K=3
128×128, Cin=16, Cout=8, K=3
256×256, Cin=16, Cout=8, K=3
512×512, Cin=16, Cout=8, K=3
```

Expected files in `exe_results/`:

```text
sequential_results_64.csv
openmp_results_64.csv
cuda_naive_results_64.csv
cuda_shared_results_64.csv
python_reference_results_64.csv
```

The script also accepts your typo:

```text
python_refernce_results_64.csv
```

Repeat the same pattern for `128`, `256`, and `512`.

## Main plots to include in report

1. `execution_time_vs_image_size.png`
2. `speedup_vs_image_size.png`
3. `cuda_naive_vs_shared_kernel_time.png`
4. `roofline_cuda.png`
5. `gflops_vs_image_size.png`

## Optional plots

1. `cuda_total_time_vs_image_size.png`
2. `python_reference_time.png`

## Timing rule

For CPU and OpenMP, the script uses `time_ms`.

For CUDA, the script uses `kernel_time_ms` for performance plots.

CUDA `total_time_ms` includes file I/O, memory transfer, allocation, and other overheads.
