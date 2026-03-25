#!/usr/bin/env zsh
#SBATCH -c 10
#SBATCH -J Sweep4
#SBATCH --partition=instruction
#SBATCH -o sweep4.out -e sweep4.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge
module load nvidia/cuda/13.0.0

clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm 61 -march=native -o task4 convolve.cpp task4.cpp

> task4_data.txt

for n in 512 1024 2048 4096
do
  ./task4 $n >> task4_data.txt
done
