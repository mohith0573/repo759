# How to Run CUDA Shared-Memory Convolution on Euler

From the project root:

```bash
cd cuda_shared
cp ../inputs/input.csv ./input.csv
cp ../inputs/kernel.csv ./kernel.csv
sbatch cuda_shared_job.sh
```

Check the job:

```bash
squeue -u $USER
```

After the job finishes:

```bash
cat cuda_shared_results.csv
cat cuda_shared.err
cat cuda_shared.out
ls cuda_shared_filter_*.csv
```

Expected checksum should match sequential. For your current 64x64 case, it should be close to:

```text
284.266132
```


