# ME759 Final Project - Software Simulation of Systolic-Array CNN Convolution

**Student:** Mohith Thatikonda  
**Course:** Spring 2026 ME/CS/ECE 759  
**Project title:** Software Simulation of Systolic-Array CNN Convolution - Sequential, OpenMP, and CUDA  
**Repository:** https://github.com/mohith0573/repo759.git

## Project goal

This project implements and evaluates a software simulation of a systolic-array-inspired CNN convolution dataflow. The model maps an `N x N` image to an `N x N` processing-element grid. Each PE computes one output pixel. Within each PE, multiple independent accumulators represent multiple MAC units, one per output filter. Each output value accumulates products across all input channels and all kernel elements.

All implementations use the same forward, top-left anchored convolution rule:

```text
output[co][h][w] = sum input[ci][h + kh][w + kw] * kernel[co][ci][kh][kw]
```

Padding is applied only when `h + kh >= H` or `w + kw >= W`, so padding is only on the bottom and right image boundaries.

The project includes five implementations:

1. Sequential C++ baseline
2. OpenMP CPU parallel implementation
3. CUDA naive global-memory implementation
4. CUDA shared-memory tiled implementation
5. Python reference implementation for correctness validation

No ML libraries such as PyTorch, cuDNN, or cuBLAS are used for the convolution logic.

## Directory structure

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

## Folder overview

### `inputs/`

Generates and stores shared input image data and kernel weights.

Example generated files:

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

Each implementation folder expects the active files to be named exactly:

```text
input.csv
kernel.csv
```

For each experiment, the appropriate size-specific files are copied from `inputs/` into each implementation folder.

---

### `sequential_execution/`

Contains the sequential C++ implementation.

Main files include:

```text
conv_sequential.cpp
conv_sequential.hpp
main_sequential.cpp
Makefile
sequential_job.sh
README_SEQ.md
HOW_TO_RUN_SEQ.md
```

The output timing file is:

```text
sequential_results.csv
```

When output matrices are enabled, the folder also generates:

```text
sequential_filter_0.csv
...
sequential_filter_7.csv
```

---

### `openmp/`

Contains the OpenMP CPU implementation.

Main files include:

```text
conv_openmp.cpp
conv_openmp.hpp
main_openmp.cpp
Makefile
openmp_job.sh
README_OPENMP.md
HOW_TO_RUN_OPENMP.md
```

The output timing file is:

```text
openmp_results.csv
```

The OpenMP version parallelizes output-pixel computation and uses private per-thread accumulators to avoid data races.

---

### `cuda_naive/`

Contains the CUDA global-memory implementation.

Main files include:

```text
conv_cuda_naive.cu
conv_cuda_naive.hpp
main_cuda_naive.cu
Makefile
cuda_naive_job.sh
README_CUDA_NAIVE.md
HOW_TO_RUN_CUDA_NAIVE.md
```

The output timing file is:

```text
cuda_naive_results.csv
```

One CUDA thread represents one PE/output pixel. Each thread computes all output filters for that pixel using private accumulators. This version reads input and kernel values directly from global memory.

---

### `cuda_shared/`

Contains the CUDA shared-memory tiled implementation.

Main files include:

```text
conv_cuda_shared.cu
conv_cuda_shared.hpp
main_cuda_shared.cu
Makefile
cuda_shared_job.sh
README_CUDA_SHARED.md
HOW_TO_RUN_CUDA_SHARED.md
```

The output timing file is:

```text
cuda_shared_results.csv
```

This version cooperatively loads a tile plus right/bottom halo into `__shared__` memory so neighboring PEs can reuse overlapping input pixels.

---

### `python_ref/`

Contains the Python reference implementation.

Main files include:

```text
reference.py
python_job.sh
README_PYTHON.md
HOW_TO_RUN_PYTHON.md
```

The output timing file is:

```text
python_reference_results.csv
```

This implementation is used for correctness validation, not optimized benchmarking.

---

### `results_comparison/`

Contains scripts and outputs for correctness validation.

This folder compares every output filter matrix from:

```text
sequential_execution/
openmp/
cuda_naive/
cuda_shared/
```

against:

```text
python_ref/
```

The comparison checks:

```text
max_abs_diff
mean_abs_diff
rmse
mismatch_count_above_tol
```

The comparison also verifies that all implementation folders used identical `input.csv` and `kernel.csv` files.

Important files:

```text
compare_all_with_python_reference.py
comparison_job.sh
comparison_report.txt
comparison_summary.csv
input_kernel_consistency.csv
```

---

### `exe_results/`

Stores benchmark CSV files for multiple image sizes.

Current benchmark configuration:

```text
Image sizes: 64x64, 128x128, 256x256, 512x512
Cin        : 16
Cout       : 8
Kernel     : 3x3
```

Example files:

```text
sequential_results_64.csv
openmp_results_64.csv
cuda_naive_results_64.csv
cuda_shared_results_64.csv
python_reference_results_64.csv
```

and similarly for 128, 256, and 512.

---

### `plots/`

Contains plotting scripts, OpenMP sweep scripts, and generated PDF plots.

Important files include:

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

Generated plots include:

```text
benchmark_summary_table.pdf
execution_time_by_image_size.pdf
speedup_by_image_size.pdf
cuda_naive_vs_shared_by_image_size.pdf
openmp_time_vs_threads.pdf
openmp_strong_scaling.pdf
openmp_efficiency.pdf
roofline_style_cuda.pdf
```

## How to run the project

### 1. Generate shared inputs

```bash
cd inputs
sbatch generate_data_job.sh
```

Then copy the generated files to the implementation folders as needed. For example:

```bash
cp inputs/input_128.csv sequential_execution/input.csv
cp inputs/kernel_128.csv sequential_execution/kernel.csv
```

Repeat for `openmp/`, `cuda_naive/`, `cuda_shared/`, and `python_ref/`.

### 2. Run each implementation

Sequential:

```bash
cd sequential_execution
sbatch sequential_job.sh
```

OpenMP:

```bash
cd openmp
sbatch openmp_job.sh
```

CUDA naive:

```bash
cd cuda_naive
sbatch cuda_naive_job.sh
```

CUDA shared:

```bash
cd cuda_shared
sbatch cuda_shared_job.sh
```

Python reference:

```bash
cd python_ref
sbatch python_job.sh
```

### 3. Run correctness comparison

```bash
cd results_comparison
sbatch comparison_job.sh
```

Check:

```bash
cat comparison_report.txt
cat comparison_summary.csv
cat input_kernel_consistency.csv
```

A successful run ends with:

```text
FINAL RESULT: PASS
```

### 4. Generate plots

```bash
cd plots
sbatch plots_job.sh
```

For OpenMP thread-scaling plots:

```bash
sbatch openmp_sweep_job.sh
sbatch plot_openmp_sweep_job.sh
```

## Timing convention

For CPU methods and Python reference, the reported value is:

```text
time_ms
```

For CUDA methods, the primary kernel performance value is:

```text
kernel_time_ms
```

CUDA `total_time_ms` includes allocation, host-device memory copies, file I/O, and output handling, so it is reported separately.

## Correctness summary

Correctness was validated by comparing all output filter matrices against the Python reference. The validation case currently stored in `results_comparison/` uses:

```text
H    = 128
W    = 128
Cin  = 16
Cout = 8
K    = 3
```

All implementations passed element-wise comparison with tolerance `1e-4`.

## Notes for the grader

- All implementation folders are independent and include their own compile/run scripts.
- Input data and kernel weights are shared through the `inputs/` folder.
- Correctness validation and performance benchmarking are separated intentionally.
- The project uses SLURM scripts compatible with the Euler/CHTC instruction partition.
- CUDA scripts request one GPU using `--gres=gpu:1`.
- The final report summarizes implementation, correctness, benchmark results, and analysis.
