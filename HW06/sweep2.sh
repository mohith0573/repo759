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

# compile
nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

> times_task2.txt

for ((i=10;i<=16;i++)); do
    n=$((2**i))
    ./task2 $n 1024 >> times_task2.txt
done
