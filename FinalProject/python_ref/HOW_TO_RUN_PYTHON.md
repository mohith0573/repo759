# How to Run Python Reference on Euler

This directory is flat. All files stay inside `python/`.

## 1. Copy shared input and kernel from sequential

From inside the `python/` directory:

```bash
cp ../inputs/input.csv ./input.csv
cp ../inputs/kernel.csv ./kernel.csv
```

These must be the same files used by sequential, OpenMP, CUDA naive, and CUDA shared.

## 2. Submit Python reference job

```bash
sbatch python_reference_job.sh
```

## 3. Check output

```bash
cat python_reference_results.csv
ls python_reference_filter_*.csv
```

For `Cout=8`, the output matrices are:

```text
python_reference_filter_0.csv
python_reference_filter_1.csv
...
python_reference_filter_7.csv
```


```

## Notes

Python reference is mainly for correctness validation, not performance comparison. It is expected to be slower than C++ and CUDA.
