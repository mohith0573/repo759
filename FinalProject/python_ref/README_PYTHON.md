# Python Reference Implementation

This folder contains a pure Python reference implementation for the project convolution.

## Convolution rule

For each output filter `co` and output pixel `(h,w)`:

```text
output[co][h][w] = sum input[ci][h+kh][w+kw] * kernel[co][ci][kh][kw]
```

Boundary rule:

```text
if h+kh >= H or w+kw >= W, use zero padding
```

So the convolution looks only to the right and bottom of the current PE/output pixel.

## Files

```text
reference.py                  Python reference implementation
python_reference_job.sh        Euler SLURM job script
compare_with_sequential.py     Numeric comparison against sequential output
README_PYTHON.md               This file
HOW_TO_RUN_PYTHON.md           Step-by-step Euler instructions
```

## Input files required

This folder expects:

```text
input.csv
kernel.csv
```

Copy them from your sequential folder:

```bash
cp ../inputs/input.csv ./input.csv
cp ../inputs/kernel.csv ./kernel.csv
```

## Output files generated

```text
python_reference_results.csv
python_reference_filter_0.csv
python_reference_filter_1.csv
...
```

## Result CSV columns

```csv
method,H,W,Cin,Cout,K,repeats,time_ms,total_time_ms,checksum,input_file,kernel_file
```

`time_ms` is the average Python execution time per repeat.
`checksum` is a quick numerical summary of the output.

