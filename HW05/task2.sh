#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task2
#SBATCH --partition=instruction
#SBATCH -o task2.out -e task2.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0
nvcc task2.cu reduce.cu -O3 -std=c++17 -o task2
./task2 1048576 1024
