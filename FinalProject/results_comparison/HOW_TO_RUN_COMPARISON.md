# How to Run Results Comparison on Euler

This folder compares all implementation output matrices against the Python reference.

It assumes this project layout:

```text
inputs/
sequential_execution/
openmp/
cuda_naive/
cuda_shared/
python_ref/
results_comparison/
```

The comparison is currently configured for:

```text
H    = 64
W    = 64
Cin  = 3
Cout = 8
K    = 3
```

## 1. Make sure all folders use the same input files

From your main project directory:

```bash
cp inputs/input.csv sequential_execution/input.csv
cp inputs/kernel.csv sequential_execution/kernel.csv

cp inputs/input.csv openmp/input.csv
cp inputs/kernel.csv openmp/kernel.csv

cp inputs/input.csv cuda_naive/input.csv
cp inputs/kernel.csv cuda_naive/kernel.csv

cp inputs/input.csv cuda_shared/input.csv
cp inputs/kernel.csv cuda_shared/kernel.csv

cp inputs/input.csv python_ref/input.csv
cp inputs/kernel.csv python_ref/kernel.csv
```

## 2. Make sure all implementations have generated output matrices

Expected files:

```text
sequential_execution/sequential_filter_0.csv ... sequential_filter_7.csv
openmp/openmp_filter_0.csv ... openmp_filter_7.csv
cuda_naive/cuda_naive_filter_0.csv ... cuda_naive_filter_7.csv
cuda_shared/cuda_shared_filter_0.csv ... cuda_shared_filter_7.csv
python_ref/python_reference_filter_0.csv ... python_reference_filter_7.csv
```

## 3. Submit the comparison job

From the main project directory:

```bash
cd results_comparison
sbatch comparison_job.sh
```

## 4. Check results

After the job finishes:

```bash
cat comparison_report.txt
cat comparison_summary.csv
cat input_kernel_consistency.csv
```

You want:

```text
FINAL RESULT: PASS
```

## 5. Manual run without SLURM

Inside `results_comparison/`:

```bash
python3 compare_all_with_python_reference.py \
    --H 64 \
    --W 64 \
    --Cin 3 \
    --Cout 8 \
    --K 3 \
    --tol 1e-4
```

## 6. If you change dimensions

If you regenerate inputs with a different size, update the command.

Example for 128 × 128:

```bash
python3 compare_all_with_python_reference.py \
    --H 128 \
    --W 128 \
    --Cin 3 \
    --Cout 8 \
    --K 3 \
    --tol 1e-4
```

If `Cout` changes, update `--Cout` also.
