#!/usr/bin/env zsh
#SBATCH -J Task4
#SBATCH --partition=instruction
#SBATCH -o task4.out -e task4.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge
module load nvidia/cuda/13.0.0

clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm 61 -march=native -o task4 convolve.cpp task4.cpp

./task4 1024
