#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1
#SBATCH --partition=instruction
#SBATCH -o task1.out -e task1.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0
nvcc task1.cu matmul.cu -O3 -std=c++17 -o task1
./task1 512 16
