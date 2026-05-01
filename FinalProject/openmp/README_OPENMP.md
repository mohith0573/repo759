# OpenMP Forward Convolution

Flat OpenMP implementation for the ME759 systolic-style CNN convolution project.

## Files

- `conv_openmp.hpp`
- `conv_openmp.cpp`
- `main_openmp.cpp`
- `Makefile`
- `openmp_job.sh`
- `compare_with_sequential.py`

## Data files required

Before running, copy the same input and kernel files used by the sequential task:

```bash
cp ../inputs/input.csv input.csv
cp ../inputs/kernel.csv kernel.csv
```

## Run on Euler

```bash
sbatch openmp_job.sh
cat openmp_results.csv
```

## Compare with sequential output


```
