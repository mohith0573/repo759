#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task3_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep.out -e sweep.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0

# Compile 512-thread version
nvcc task3.cu vscale.cu -DTPB=512 -Xcompiler -O3 -std=c++17 -o task3_512

# Compile 16-thread version
nvcc task3.cu vscale.cu -DTPB=16 -Xcompiler -O3 -std=c++17 -o task3_16

echo "n time_512 time_16" > timing.txt

for i in {10..29}
do
    n=$((2**i))

    t512=$(./task3_512 $n | head -n 1)
    t16=$(./task3_16 $n | head -n 1)

    echo "$n $t512 $t16" >> timing.txt
done

