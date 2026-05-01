# How to Run Results Comparison on Euler

This folder compares all implementation output matrices with the Python reference output matrices.

The comparison is for:

```text
H    = 64
W    = 64
Cin  = 3
Cout = 8
K    = 3
```

## 1. Make sure all implementations have already been run

You should already have generated these files:

```text
sequential_implementation/sequential_filter_0.csv ... sequential_filter_7.csv
openmp/openmp_filter_0.csv ... openmp_filter_7.csv
cuda_naive/cuda_naive_filter_0.csv ... cuda_naive_filter_7.csv
cuda_shared/cuda_shared_filter_0.csv ... cuda_shared_filter_7.csv
python_ref/python_reference_filter_0.csv ... python_reference_filter_7.csv
```

Also make sure each folder contains the same:

```text
input.csv
kernel.csv
```

These files should be copied from the `inputs/` folder.

Example:

```bash
cp inputs/input.csv sequential_implementation/input.csv
cp inputs/kernel.csv sequential_implementation/kernel.csv

cp inputs/input.csv openmp/input.csv
cp inputs/kernel.csv openmp/kernel.csv

cp inputs/input.csv cuda_naive/input.csv
cp inputs/kernel.csv cuda_naive/kernel.csv

cp inputs/input.csv cuda_shared/input.csv
cp inputs/kernel.csv cuda_shared/kernel.csv

cp inputs/input.csv python_ref/input.csv
cp inputs/kernel.csv python_ref/kernel.csv
```

## 2. Go into the comparison folder

From the main project directory:

```bash
cd results_comparison
```

## 3. Submit comparison job

```bash
sbatch comparison_job.sh
```

## 4. Check output

After the job finishes:

```bash
cat comparison_report.txt
cat comparison_summary.csv
cat input_kernel_consistency.csv
```

You want to see:

```text
FINAL RESULT: PASS
```

## 5. Manual run without SLURM

You can also run directly:

```bash
python3 compare_all_with_python_reference.py \
    --H 64 \
    --W 64 \
    --Cin 3 \
    --Cout 8 \
    --K 3 \
    --tol 1e-4
```

## 6. If you change problem size

If you regenerate inputs with different dimensions, update the command arguments.

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

If you change `Cout`, you must also change `--Cout`.

For example, if `Cout = 16`, use:

```bash
--Cout 16
```
