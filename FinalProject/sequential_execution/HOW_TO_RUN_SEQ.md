# How to Run Sequential Task on Euler

This folder is flat. Run all commands from inside this `seq` directory.

## 1. Generate shared input image data and weight data

Submit the data generation job:

```bash
sbatch generate_data_job.sh
```

After it finishes, check:

```bash
ls input.csv kernel.csv
cat generate_data.out
```

These two files are the shared data files you can copy later into `openmp`, `cuda_naive`, `cuda_shared`, and `python`.

## 2. Run sequential implementation

Submit the sequential job:

```bash
sbatch sequential_job.sh
```

After it finishes, check timing:

```bash
cat sequential_results.csv
```

Check output matrices:

```bash
ls sequential_filter_*.csv
```

For `Cout=8`, you should see:

```text
sequential_filter_0.csv
sequential_filter_1.csv
...
sequential_filter_7.csv
```

## 3. Manual run without SLURM

```bash
python3 generate_data.py --H 64 --W 64 --Cin 3 --Cout 8 --K 3 --input input.csv --kernel kernel.csv
make clean
make
./conv_sequential 64 64 3 8 3 5 input.csv kernel.csv 1 > sequential_results.csv
```

## 4. Meaning of repeats

`repeats` is only for timing stability.

Example:

```bash
./conv_sequential 64 64 3 8 3 5 input.csv kernel.csv 1
```

Here `repeats = 5`, so the convolution runs 5 times and the program reports the average time:

```text
time_ms = total_time_ms / repeats
```

It does not change the mathematical output. It just gives a more stable execution-time measurement.

## 5. If you change dimensions

If you generate data with different dimensions, update the same values in `sequential_job.sh`.

For example, if you generate:

```bash
python3 generate_data.py --H 128 --W 128 --Cin 8 --Cout 8 --K 3
```

then update this part of `sequential_job.sh`:

```bash
H=128
W=128
Cin=8
Cout=8
K=3
```
