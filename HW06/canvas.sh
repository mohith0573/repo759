#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Cuda-Memcheck
#SBATCH --partition=instruction
#SBATCH -o canvas.out -e canvas.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

cuda-memcheck ./task2 1024 1024
