# Sequential Forward Convolution Task

This folder is intentionally flat. All files stay in this same directory.

## Files

- `generate_data.py` : generates shared `input.csv` and `kernel.csv`
- `conv_sequential.hpp` : function declaration
- `conv_sequential.cpp` : locality-friendly sequential convolution implementation
- `main_sequential.cpp` : reads input/kernel CSV files, times execution, writes output matrices
- `generate_data_job.sh` : SLURM job to generate data on Euler
- `sequential_job.sh` : SLURM job to compile and run sequential implementation
- `Makefile` : optional manual build helper

## Convolution rule

For each output pixel `(h,w)` and output filter `co`:

```text
output[co][h][w] = sum input[ci][h+kh][w+kw] * kernel[co][ci][kh][kw]
```

Padding is only on the bottom and right boundaries.

## Data layout

`input.csv` contains values in this order:

```text
input[ci][h][w]
index = (ci * H + h) * W + w
```

`kernel.csv` contains values in this order:

```text
kernel[co][ci][kh][kw]
index = ((co * Cin + ci) * K + kh) * K + kw
```

Output files are written as:

```text
sequential_filter_0.csv
sequential_filter_1.csv
...
```

Each output file is one H x W matrix for one output filter.
