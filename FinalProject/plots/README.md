# ME759 Final Project: Plots and OpenMP Sweep

This folder contains the plotting and OpenMP thread-sweep utilities for the final project:

```text
Software Simulation of Systolic-Array CNN Convolution
Sequential, OpenMP, CUDA Naive, CUDA Shared Memory, and Python Reference
```

This project evaluates a forward, top-left anchored 2D convolution model inspired by a systolic PE-array dataflow. Each output pixel is treated as one PE. For each PE, multiple output filters are computed using independent accumulators, corresponding to the multi-MAC-per-PE model.

The convolution rule used by all implementations is:

```text
output[co][h][w] =
    sum over ci, kh, kw of input[ci][h + kh][w + kw] * kernel[co][ci][kh][kw]
```

Padding is applied only when `h + kh >= H` or `w + kw >= W`, so the padding is on the bottom and right boundaries.

---

## Expected FinalProject directory structure

This folder is intended to be placed inside the main project directory:

```text
FinalProject/
├── inputs/
├── sequential_execution/
├── openmp/
├── cuda_naive/
├── cuda_shared/
├── python_ref/
├── results_comparison/
├── exe_results/
└── plots/
```

This `plots/` folder contains scripts for:

1. Generating performance plots from existing timing CSV files in `exe_results/`.
2. Running an OpenMP thread-count sweep.
3. Plotting OpenMP strong-scaling results.

---

## Important data and correctness workflow

The project uses one shared input and one shared kernel per image size.

Example files in `inputs/`:

```text
input_64.csv
kernel_64.csv
input_128.csv
kernel_128.csv
input_256.csv
kernel_256.csv
input_512.csv
kernel_512.csv
```

Each implementation folder uses files named exactly:

```text
input.csv
kernel.csv
```

For each run, the matching size-specific files from `inputs/` are copied into the implementation folder as `input.csv` and `kernel.csv`.

Correctness comparison is handled separately in:

```text
results_comparison/
```

That folder compares every output filter matrix from all methods against the Python reference output using element-wise numerical tolerance.

---

## Benchmark result files

The main benchmark timing CSV files are stored in:

```text
exe_results/
```

Expected examples:

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
python_reference_results_128.csv

sequential_results_256.csv
openmp_results_256.csv
cuda_naive_results_256.csv
cuda_shared_results_256.csv
python_reference_results_256.csv

sequential_results_512.csv
openmp_results_512.csv
cuda_naive_results_512.csv
cuda_shared_results_512.csv
python_reference_results_512.csv
```

The current benchmark configuration is:

```text
Image sizes: 64x64, 128x128, 256x256, 512x512
Cin        : 16
Cout       : 8
Kernel     : 3x3
```

The large benchmark runs use:

```text
WRITE_MATRICES = 0
```

because correctness was already checked separately and writing full output matrices can dominate runtime.

---

## Plot scripts in this folder

This folder may contain the following files:

```text
generate_all_plots.py
plots_job.sh

openmp_sweep_job.sh
plot_openmp_sweep.py
plot_openmp_sweep_job.sh

README.md
HOW_TO_RUN_PLOTS_AND_OPENMP_SWEEP.md
REPORT_PLOTS_AND_SWEEP_NOTES.md
```

### Main benchmark plotting

`generate_all_plots.py` reads timing CSV files from:

```text
../exe_results/
```

and generates PDF plots in this folder.

Generated files include:

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

### OpenMP sweep

`openmp_sweep_job.sh` runs the OpenMP implementation for multiple thread counts and image sizes.

Default sweep configuration:

```text
Image sizes: 64, 128, 256, 512
Cin        : 16
Cout       : 8
K          : 3
Threads    : 1, 2, 4, 8, 16, 20
Repeats    : 5
```

It writes sweep CSVs into:

```text
../exe_results/
```

Generated sweep files:

```text
openmp_sweep_64.csv
openmp_sweep_128.csv
openmp_sweep_256.csv
openmp_sweep_512.csv
openmp_sweep_all.csv
```

`plot_openmp_sweep.py` then generates:

```text
openmp_time_vs_threads.pdf
openmp_strong_scaling.pdf
openmp_efficiency.pdf
openmp_scaling_summary.csv
```

---

## Timing convention

For CPU methods:

```text
time_ms
```

is used.

For CUDA methods:

```text
kernel_time_ms
```

is used for kernel performance plots.

CUDA `total_time_ms` is not the primary performance number because it includes CPU/GPU memory allocation, host-device copies, file I/O, and output handling. It is still plotted separately in:

```text
total_time_by_image_size.pdf
```

---

## Notes for the grader

The plotting workflow is intentionally separated from correctness validation.

Correctness validation:

```text
results_comparison/
```

Performance benchmarking:

```text
exe_results/
```

Plot generation:

```text
plots/
```

The main benchmark plots show how runtime and speedup vary with image size. The OpenMP sweep plots show how the OpenMP implementation scales with thread count.
