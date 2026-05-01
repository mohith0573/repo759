# How to Run CUDA Naive on Euler

Put this `cuda_naive/` folder beside your other project folders:

```text
repo759/
├── seq/
├── openmp/
├── cuda_naive/
├── cuda_shared/
└── python/
```

## 1. Copy the same input and kernel files

From inside `cuda_naive/`:

```bash
cp ../inputs/input.csv ./input.csv
cp ../inputs/kernel.csv ./kernel.csv
```

These files must be the same files used for sequential and OpenMP.

## 2. Submit the CUDA naive job

```bash
sbatch cuda_naive_job.sh
```

The script uses the course-style module command:

```bash
module load nvidia/cuda/13.0.0
```

## 3. Check timing results

```bash
cat cuda_naive_results.csv
```

The output format is:

```csv
method,H,W,Cin,Cout,K,threads_per_block,repeats,kernel_time_ms,total_time_ms,checksum,input_file,kernel_file,gpu_name
```

Use `kernel_time_ms` for CUDA kernel performance.

## 4. Check output matrices

```bash
ls cuda_naive_filter_*.csv
```

For `Cout=8`, you should get:

```text
cuda_naive_filter_0.csv
cuda_naive_filter_1.csv
...
cuda_naive_filter_7.csv
```



Arguments:

```text
./conv_cuda_naive H W Cin Cout K repeats input.csv kernel.csv write_matrices
```

`write_matrices=1` writes output matrices. Use `0` for large benchmarking runs.
