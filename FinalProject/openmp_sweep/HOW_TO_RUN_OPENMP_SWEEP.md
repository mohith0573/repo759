# How to Run OpenMP Sweep on Euler

Your main project directory is:

```text
FinalProject/
```

This folder should be located at:

```text
FinalProject/openmp_sweep/
```

## 1. Go to the sweep folder

```bash
cd FinalProject/openmp_sweep
```

## 2. Run the OpenMP sweep

```bash
sbatch openmp_sweep_job.sh
```

Check progress/output:

```bash
cat openmp_sweep.out
cat openmp_sweep.err
```

## 3. Check generated CSV results

After the job finishes:

```bash
ls ../exe_results/openmp_sweep_*.csv
cat ../exe_results/openmp_sweep_all.csv
```

You should see timing rows for:

```text
threads = 1, 2, 4, 8, 16, 20
```

for each image size:

```text
64, 128, 256, 512
```

## 4. Generate OpenMP scaling plots

```bash
sbatch plot_openmp_sweep_job.sh
```

After it finishes:

```bash
ls *.pdf
cat plot_openmp_sweep.out
cat plot_openmp_sweep.err
```

Generated plots:

```text
openmp_time_vs_threads.pdf
openmp_strong_scaling.pdf
openmp_efficiency.pdf
```

## 5. Manual plotting without SLURM

```bash
python3 plot_openmp_sweep.py
```

## 6. Change benchmark configuration

Edit this part of `openmp_sweep_job.sh`:

```bash
SIZES=(64 128 256 512)
Cin=16
Cout=8
K=3
REPEATS=5
THREAD_LIST=(1 2 4 8 16 20)
```

If your inputs change, make sure the files in `../inputs/` match the selected sizes and dimensions.
