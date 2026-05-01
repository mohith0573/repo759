# How to Generate Input and Kernel Files on Euler

This file explains how to generate the common input image data and convolution weight data for the project.

The generated files are:

```text
input.csv
kernel.csv
```

These files must stay in CSV format and must use these exact names.

---

## Step 1: Go to the `inputs` folder

From the main project directory:

```bash
cd inputs
```

Your folder should contain something like:

```text
generate_data.py
generate_data_job.sh
README.md
HOW_TO_GENERATE_INPUTS.md
```

---

## Step 2: Generate input data using SLURM

Submit the input-generation job:

```bash
sbatch generate_data_job.sh
```

After the job finishes, check that the files were created:

```bash
ls input.csv kernel.csv
```

You should see:

```text
input.csv
kernel.csv
```

---

## Step 3: Check the job output

```bash
cat generate_data.out
cat generate_data.err
```

If `generate_data.err` is empty, that is usually good.

---

## Step 4: Copy the same files to every task folder

From the main project directory:

```bash
cd ..

cp inputs/input.csv sequential_implementation/input.csv
cp inputs/kernel.csv sequential_implementation/kernel.csv

cp inputs/input.csv openmp/input.csv
cp inputs/kernel.csv openmp/kernel.csv

cp inputs/input.csv cuda_naive/input.csv
cp inputs/kernel.csv cuda_naive/kernel.csv

cp inputs/input.csv cuda_shared/input.csv
cp inputs/kernel.csv cuda_shared/kernel.csv

cp inputs/input.csv python/input.csv
cp inputs/kernel.csv python/kernel.csv
```

Now every implementation uses the same image data and the same weight data.

---

## Step 5: Run each implementation

After copying the files, run each implementation from its own folder.

Example:

```bash
cd sequential_implementation
sbatch sequential_job.sh
```

```bash
cd ../openmp
sbatch openmp_job.sh
```

```bash
cd ../cuda_naive
sbatch cuda_naive_job.sh
```

```bash
cd ../cuda_shared
sbatch cuda_shared_job.sh
```

```bash
cd ../python
sbatch python_reference_job.sh
```

---

## Changing input specifications

You can change the image size, number of channels, number of filters, and kernel size.

For example:

```text
H = 64
W = 64
Cin = 3
Cout = 8
K = 3
```

means:

```text
input.csv  contains 3 input channels, each 64×64
kernel.csv contains 8 filters, each with 3 kernel slices of size 3×3
```

So:

```text
input.csv values  = Cin × H × W
                  = 3 × 64 × 64
                  = 12,288 values

kernel.csv values = Cout × Cin × K × K
                  = 8 × 3 × 3 × 3
                  = 216 values
```

If you change these values in `generate_data_job.sh`, you must also update the same values in every implementation job script:

```text
seq/sequential_job.sh
openmp/openmp_job.sh
cuda_naive/cuda_naive_job.sh
cuda_shared/cuda_shared_job.sh
python/python_reference_job.sh
```

The values must match exactly:

```text
H, W, Cin, Cout, K
```

If they do not match, the program may read the wrong number of values or produce incorrect results.

---

## Example `generate_data_job.sh`

```bash
#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J GenInput
#SBATCH --partition=instruction
#SBATCH -o generate_data.out
#SBATCH -e generate_data.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module purge

H=64
W=64
Cin=3
Cout=8
K=3

python3 generate_data.py \
    --H $H \
    --W $W \
    --Cin $Cin \
    --Cout $Cout \
    --K $K \
    --input input.csv \
    --kernel kernel.csv
```

---

## Correctness requirement

For correctness comparison, all implementations must use the same copied files:

```text
input.csv
kernel.csv
```

After running all implementations, their checksums should match or be extremely close.

Expected pattern:

```text
sequential checksum        = same value
openmp checksum            = same value
cuda_naive checksum        = same value
cuda_shared checksum       = same value
python_reference checksum  = same value
```

Small floating-point differences are acceptable, but the outputs should pass element-wise comparison with a tolerance such as:

```text
1e-4
```
