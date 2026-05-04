# How to Run Plots and OpenMP Sweep on Euler

This file explains how to use the combined plotting and OpenMP sweep folder.

The main project directory is assumed to be:

```text
FinalProject/
```

The plotting folder is:

```text
FinalProject/plots/
```

---

## 1. Go to the plots folder

```bash
cd FinalProject/plots
```

---

# Part A: Generate main benchmark plots

The main benchmark plots use existing timing CSV files from:

```text
FinalProject/exe_results/
```

These CSV files should already contain timing results for:

```text
64x64
128x128
256x256
512x512
```

with:

```text
Cin  = 16
Cout = 8
K    = 3
```

## Run plotting job

```bash
sbatch plots_job.sh
```

## Check output

After the job finishes:

```bash
cat plots.out
cat plots.err
ls *.pdf
```

Expected PDF files:

```text
execution_time_by_image_size.pdf
total_time_by_image_size.pdf
speedup_by_image_size.pdf
cuda_naive_vs_shared_by_image_size.pdf
roofline_style_cuda.pdf
benchmark_summary_table.pdf
```

Expected CSV files:

```text
combined_benchmark_results.csv
roofline_cuda_table.csv
```

## Manual run without SLURM

```bash
python3 generate_all_plots.py
```

---

# Part B: Run OpenMP thread-count sweep

The OpenMP sweep is separate from the main image-size benchmark.

It runs the OpenMP implementation with different thread counts:

```text
1, 2, 4, 8, 16, 20
```

for image sizes:

```text
64, 128, 256, 512
```

The default configuration inside `openmp_sweep_job.sh` is:

```text
Cin  = 16
Cout = 8
K    = 3
```

## Required input files

The script expects size-specific input files in:

```text
FinalProject/inputs/
```

Expected names:

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

The script also supports names without `.csv`:

```text
input_64
kernel_64
...
```

For each image size, the script copies the selected files into:

```text
FinalProject/openmp/input.csv
FinalProject/openmp/kernel.csv
```

## Submit OpenMP sweep job

From inside `FinalProject/plots/`:

```bash
sbatch openmp_sweep_job.sh
```

## Check sweep output

```bash
cat openmp_sweep.out
cat openmp_sweep.err
ls ../exe_results/openmp_sweep_*.csv
```

Expected output files:

```text
../exe_results/openmp_sweep_64.csv
../exe_results/openmp_sweep_128.csv
../exe_results/openmp_sweep_256.csv
../exe_results/openmp_sweep_512.csv
../exe_results/openmp_sweep_all.csv
```

---

# Part C: Generate OpenMP sweep plots

After `openmp_sweep_job.sh` finishes, run:

```bash
sbatch plot_openmp_sweep_job.sh
```

Check output:

```bash
cat plot_openmp_sweep.out
cat plot_openmp_sweep.err
ls openmp_*.pdf
```

Expected files:

```text
openmp_time_vs_threads.pdf
openmp_strong_scaling.pdf
openmp_efficiency.pdf
openmp_scaling_summary.csv
```

## Manual OpenMP sweep plotting

```bash
python3 plot_openmp_sweep.py
```

---

# Part D: Recommended order for final report figures

Use this order:

```text
1. benchmark_summary_table.pdf
2. execution_time_by_image_size.pdf
3. speedup_by_image_size.pdf
4. cuda_naive_vs_shared_by_image_size.pdf
5. openmp_time_vs_threads.pdf
6. openmp_strong_scaling.pdf
7. openmp_efficiency.pdf
8. roofline_style_cuda.pdf
```

`total_time_by_image_size.pdf` is optional. Use it only if discussing end-to-end overhead.

---

# Part E: If dimensions change

If a new benchmark uses different parameters, update the corresponding script variables.

For OpenMP sweep, edit:

```bash
SIZES=(64 128 256 512)
Cin=16
Cout=8
K=3
REPEATS=5
THREAD_LIST=(1 2 4 8 16 20)
```

inside:

```text
openmp_sweep_job.sh
```

For main plots, no parameter change is usually needed because `generate_all_plots.py` reads values from the CSV files.
