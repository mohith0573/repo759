#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep1.out -e sweep1.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0

nvcc task1.cu matmul.cu -O3 -std=c++17 -o task1

for ((i=5;i<=14;i++)); do
 n=$((2**i))
 ./task1 $n 16 >> times16.txt
 ./task1 $n 32 >> times32.txt
done