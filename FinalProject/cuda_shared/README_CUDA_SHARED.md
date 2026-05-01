# CUDA Shared-Memory Tiled Convolution

This folder contains the CUDA shared-memory implementation for the ME759 final project.

It uses the same forward/right-bottom convolution rule as the sequential, OpenMP, and CUDA naive versions:

```text
output[co][h][w] += input[ci][h + kh][w + kw] * kernel[co][ci][kh][kw]
```

If `h + kh >= H` or `w + kw >= W`, the value is treated as zero. Therefore padding is only on the bottom and right.

## Files

```text
conv_cuda_shared.hpp
conv_cuda_shared.cu
main_cuda_shared.cu
Makefile
cuda_shared_job.sh
compare_with_sequential.py
```

## Data files required

Copy the same files used by sequential:

```bash
cp ../inputs/input.csv ./input.csv
cp ../inputs/kernel.csv ./kernel.csv
```

## Run on Euler

```bash
sbatch cuda_shared_job.sh
```

## Output

```text
cuda_shared_results.csv
cuda_shared_filter_0.csv
cuda_shared_filter_1.csv
...
```

