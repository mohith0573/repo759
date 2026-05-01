# CUDA Shared-Memory Tiled Forward Convolution

This folder is intentionally flat. All files stay directly inside `cuda_shared/`.

## Files

- `conv_cuda_shared.hpp` - host wrapper declaration
- `conv_cuda_shared.cu` - CUDA shared-memory tiled kernel and host wrapper
- `main_cuda_shared.cu` - file I/O, timing, checksum, and matrix writing
- `Makefile` - builds `conv_cuda_shared`
- `cuda_shared_job.sh` - Euler SLURM script
- `compare_with_sequential.py` - numerical comparison against sequential output matrices

## Convolution rule

The implementation uses the project-specific forward/right-bottom convolution:

```text
output[co][h][w] = sum input[ci][h+kh][w+kw] * kernel[co][ci][kh][kw]
```

If `h+kh >= H` or `w+kw >= W`, the input value is treated as zero. This means padding is only on the bottom and right.

## Data layout

```text
input[ci][h][w]        index = (ci * H + h) * W + w
kernel[co][ci][kh][kw] index = ((co * Cin + ci) * K + kh) * K + kw
output[co][h][w]       index = (co * H + h) * W + w
```

## CUDA mapping

- One CUDA thread = one PE = one output pixel `(h,w)`
- Each thread maintains one accumulator per output filter
- Shared memory stores one input-channel tile plus right/bottom halo
- The tile is reused by neighboring PEs inside the same CUDA block

## Run on Euler

Copy the same input and kernel files used by sequential:

```bash
cp ../seq/input.csv ./input.csv
cp ../seq/kernel.csv ./kernel.csv
```

Submit:

```bash
sbatch cuda_shared_job.sh
```

Check:

```bash
cat cuda_shared_results.csv
ls cuda_shared_filter_*.csv
```

Compare with sequential:

```bash
python3 compare_with_sequential.py --seq-dir ../seq --cout 8 --tol 1e-4
```
