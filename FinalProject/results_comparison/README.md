# Results Comparison

This folder compares all implementation output matrices against the Python reference output.

This comparison is currently for:

```text
H    = 64
W    = 64
Cin  = 3
Cout = 8
K    = 3
```

That means:

```text
Input data  = 3 input channels, each 64 × 64
Kernel data = 8 filters × 3 input channels × 3 × 3 weights
Output data = 8 output matrices, each 64 × 64
```

## Project folder names used

This comparison folder assumes your main project directory contains:

```text
inputs/
sequential_execution/
openmp/
cuda_naive/
cuda_shared/
python_ref/
results_comparison/
```

## Required input files

Every implementation folder should contain the same two input files copied from `inputs/`:

```text
input.csv
kernel.csv
```

Copy example:

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

## Required output matrix files

Before running this comparison, each implementation should already have generated output files:

```text
sequential_execution/sequential_filter_0.csv ... sequential_filter_7.csv
openmp/openmp_filter_0.csv ... openmp_filter_7.csv
cuda_naive/cuda_naive_filter_0.csv ... cuda_naive_filter_7.csv
cuda_shared/cuda_shared_filter_0.csv ... cuda_shared_filter_7.csv
python_ref/python_reference_filter_0.csv ... python_reference_filter_7.csv
```

## What this comparison checks

The script checks two things.

### 1. Input/kernel consistency

It verifies that all folders use byte-identical:

```text
input.csv
kernel.csv
```

It writes:

```text
input_kernel_consistency.csv
```

### 2. Output correctness

For each filter, it compares:

```text
sequential_execution/sequential_filter_i.csv  vs python_ref/python_reference_filter_i.csv
openmp/openmp_filter_i.csv                    vs python_ref/python_reference_filter_i.csv
cuda_naive/cuda_naive_filter_i.csv            vs python_ref/python_reference_filter_i.csv
cuda_shared/cuda_shared_filter_i.csv          vs python_ref/python_reference_filter_i.csv
```

It computes:

```text
max_abs_diff
mean_abs_diff
rmse
number of values above tolerance
```

Default tolerance:

```text
1e-4
```

A comparison passes if:

```text
max_abs_diff <= 1e-4
```

## Generated files

After running, this folder generates:

```text
comparison_report.txt
comparison_summary.csv
input_kernel_consistency.csv
comparison.out
comparison.err
```

## Important note

Do not use plain Linux `diff` for floating-point matrix files. `diff` compares text formatting, so `1.23456000` and `1.23456` may appear different even though they are numerically the same. This folder uses numerical comparison with tolerance.
