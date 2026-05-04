# How to Generate Plots on Euler

This plotting folder does not include OpenMP thread-scaling plots.

## 1. Go to the plots folder

From the main project directory:

```bash
cd plots
```

## 2. Submit the plotting job

```bash
sbatch plots_job.sh
```

## 3. Check generated PDFs

After the job finishes:

```bash
ls *.pdf
cat plots.out
cat plots.err
```

Expected files:

```text
execution_time_by_image_size.pdf
total_time_by_image_size.pdf
speedup_by_image_size.pdf
cuda_naive_vs_shared_by_image_size.pdf
roofline_style_cuda.pdf
benchmark_summary_table.pdf
```

## 4. Manual run without SLURM

Inside `plots/`:

```bash
python3 generate_all_plots.py
```

## 5. What data is used?

The plotting script reads:

```text
../exe_results/*.csv
```

For CUDA timing, it uses:

```text
kernel_time_ms
```

For sequential, OpenMP, and Python reference timing, it uses:

```text
time_ms
```

## 6. Benchmark table

The generated PDF benchmark table is:

```text
benchmark_summary_table.pdf
```

This table can be directly included in the final report.
