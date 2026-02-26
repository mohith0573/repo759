#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task2_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep2.out -e sweep2.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0

nvcc task2.cu reduce.cu -O3 -std=c++17 -o task2

> times1024.txt
> times256.txt

for ((i=10;i<=30;i++)); do
 N=$((2**i))
 ./task2 $N 1024 >> times1024.txt
 ./task2 $N 256 >> times256.txt
done
