# How to Run CUDA Shared on Euler

From your project root, you should have:

```text
repo759/
├── seq/
├── openmp/
├── cuda_naive/
├── cuda_shared/
└── python/
```

Go to the CUDA shared directory:

```bash
cd cuda_shared
```

Copy the exact same input and kernel files from sequential:

```bash
cp ../seq/input.csv ./input.csv
cp ../seq/kernel.csv ./kernel.csv
```

Submit the job:

```bash
sbatch cuda_shared_job.sh
```

Check queue:

```bash
squeue -u $USER
```

After it finishes:

```bash
cat cuda_shared_results.csv
cat cuda_shared.err
cat cuda_shared.out
ls cuda_shared_filter_*.csv
```

Expected result file format:

```csv
method,H,W,Cin,Cout,K,threads_per_block,repeats,kernel_time_ms,total_time_ms,checksum,input_file,kernel_file,gpu_name
```

For performance comparison, use `kernel_time_ms`.

For end-to-end runtime, use `total_time_ms`, but remember this includes file reading, memory allocation, host-device copies, output writing, and cleanup.

For correctness:

```bash
python3 compare_with_sequential.py --seq-dir ../seq --cout 8 --tol 1e-4
```

Expected:

```text
FINAL RESULT: PASS
```

If you change `H`, `W`, `Cin`, `Cout`, or `K`, update the same values in `cuda_shared_job.sh`. They must match the dimensions used to generate `input.csv` and `kernel.csv`.
