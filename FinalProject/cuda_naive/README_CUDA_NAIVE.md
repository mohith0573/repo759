# CUDA Naive Forward Convolution

This directory contains the CUDA naive implementation for the ME759 final project.

All files are intentionally kept flat in this directory:

- `conv_cuda_naive.hpp`
- `conv_cuda_naive.cu`
- `main_cuda_naive.cu`
- `Makefile`
- `cuda_naive_job.sh`
- `compare_with_sequential.py`

The program reads:

- `input.csv`
- `kernel.csv`

from the same directory and writes:

- `cuda_naive_results.csv`
- `cuda_naive_filter_0.csv`, `cuda_naive_filter_1.csv`, ...

## Convolution rule

The implementation uses the forward/right-bottom convolution:

```text
output[co][h][w] = sum input[ci][h + kh][w + kw] * kernel[co][ci][kh][kw]
```

If `h + kh >= H` or `w + kw >= W`, that value is treated as zero padding.

## CUDA mapping

- One CUDA thread = one PE = one output pixel position `(h, w)`.
- Each thread keeps one accumulator per output filter.
- The same input value is reused across all filters inside the thread.
- This is still called "naive" because it reads directly from global memory and does not use shared-memory tiling.
