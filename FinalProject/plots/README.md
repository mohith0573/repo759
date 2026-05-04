# Plots and Benchmark Tables

This folder generates PDF plots and a PDF benchmark table for the ME759 final project.



It assumes main project directory contains:

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

## Benchmark data location

This folder reads benchmark CSV files from:

```text
../exe_results/
```

Expected example filenames:

```text
sequential_results_64.csv
openmp_results_64.csv
cuda_naive_results_64.csv
cuda_shared_results_64.csv
python_reference_results_64.csv

sequential_results_128.csv
openmp_results_128.csv
cuda_naive_results_128.csv
cuda_shared_results_128.csv

sequential_results_256.csv
openmp_results_256.csv
cuda_naive_results_256.csv
cuda_shared_results_256.csv

sequential_results_512.csv
openmp_results_512.csv
cuda_naive_results_512.csv
cuda_shared_results_512.csv
```

current benchmark configuration is:

```text
64x64,  16 Cin, 8 Cout, 3x3 kernel
128x128, 16 Cin, 8 Cout, 3x3 kernel
256x256, 16 Cin, 8 Cout, 3x3 kernel
512x512, 16 Cin, 8 Cout, 3x3 kernel
```

## Generated files

Running `generate_all_plots.py` creates:

```text
combined_benchmark_results.csv
execution_time_by_image_size.pdf
total_time_by_image_size.pdf
speedup_by_image_size.pdf
cuda_naive_vs_shared_by_image_size.pdf
roofline_style_cuda.pdf
roofline_cuda_table.csv
benchmark_summary_table.pdf
```

## Timing convention

For CPU and Python methods, plots use:

```text
time_ms
```

For CUDA methods, plots use:

```text
kernel_time_ms
```

CUDA `total_time_ms` includes memory transfers, file I/O, allocation, and output writing. It is plotted separately in:

```text
total_time_by_image_size.pdf
```
