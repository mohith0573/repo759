# How to Run Python Reference on Euler

This directory is flat. All files stay inside `python/`.

## 1. Copy shared input and kernel from sequential

From inside the `python/` directory:

```bash
cp ../seq/input.csv ./input.csv
cp ../seq/kernel.csv ./kernel.csv
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

## 4. Compare with sequential

After sequential output matrices exist:

```bash
python3 compare_with_sequential.py --seq-dir ../seq --cout 8 --tol 1e-4
```

Expected result:

```text
FINAL RESULT: PASS
```

## Manual run

```bash
python3 reference.py --H 64 --W 64 --Cin 3 --Cout 8 --K 3 \
  --repeats 1 --input input.csv --kernel kernel.csv \
  --write-matrices 1 --prefix python_reference > python_reference_results.csv
```

## Notes

Python reference is mainly for correctness validation, not performance comparison. It is expected to be slower than C++ and CUDA.
