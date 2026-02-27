#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_sweep_14_32
#SBATCH --partition=instruction
#SBATCH -o sweep1_14_32.out -e sweep1_14_32.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0

nvcc task1.cu matmul.cu -O3 -std=c++17 -o task1

for ((i=14;i<=14;i++)); do
 n=$((2**i))
 ./task1 $n 32 >> times32.txt
done
