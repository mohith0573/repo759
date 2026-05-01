# How to Run the OpenMP Task on Euler

## 1. Go to the OpenMP directory

```bash
cd openmp
```

## 2. Copy the same data from the sequential directory

```bash
cp ../inputs/input.csv ./input.csv
cp ../inputs/kernel.csv ./kernel.csv
```

This is important because correctness comparison only works when sequential and OpenMP use the same input image data and the same weights.

## 3. Submit the OpenMP job

```bash
sbatch openmp_job.sh
```

## 4. Check timing

```bash
cat openmp_results.csv
```

The important fields are:

- `time_ms`: average OpenMP execution time per run
- `total_time_ms`: total time for all repeated runs
- `checksum`: compact output summary used for quick comparison

## 5. Check output matrices

```bash
ls openmp_filter_*.csv
```

For `Cout=8`, the program writes:

```text
openmp_filter_0.csv
openmp_filter_1.csv
...
openmp_filter_7.csv
```



You can also compare checksums:

```bash
cat sequential_execution/sequential_results.csv
cat openmp/openmp_results.csv
```

The checksums should match or be extremely close.
