# OpenMP Sweep

This folder runs OpenMP thread-count sweeps for the ME759 final project.

Place this folder inside your main project directory:

```text
FinalProject/
├── inputs/
├── openmp/
├── exe_results/
└── openmp_sweep/
```

## What this sweep does

It runs the OpenMP implementation for multiple image sizes and multiple thread counts.

Default benchmark configuration:

```text
Image sizes: 64, 128, 256, 512
Cin        : 16
Cout       : 8
K          : 3
Repeats    : 5
Threads    : 1, 2, 4, 8, 16, 20
```

It uses `WRITE_MATRICES=0`, so it only records timing and does not write output matrices.

## Input files expected

The script reads size-specific inputs from:

```text
FinalProject/inputs/
```

It expects either:

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

or the same names without `.csv`:

```text
input_64
kernel_64
...
```

For each image size, it copies the matching input/kernel files into:

```text
FinalProject/openmp/input.csv
FinalProject/openmp/kernel.csv
```

and then runs the OpenMP executable.

## Output files

The sweep writes CSV files to:

```text
FinalProject/exe_results/
```

Generated files:

```text
openmp_sweep_64.csv
openmp_sweep_128.csv
openmp_sweep_256.csv
openmp_sweep_512.csv
openmp_sweep_all.csv
```

The combined file `openmp_sweep_all.csv` is useful for plotting.

## Generated plots

The plotting script creates:

```text
openmp_time_vs_threads.pdf
openmp_strong_scaling.pdf
openmp_efficiency.pdf
openmp_scaling_summary.csv
```
