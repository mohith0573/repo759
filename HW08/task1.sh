#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Task1
#SBATCH --partition=instruction
#SBATCH -o task1.out -e task1.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

g++ task1.cpp matmul.cpp -O3 -std=c++17 -fopenmp -o task1
./task1 1024 8
