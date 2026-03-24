#!/usr/bin/env zsh
#SBATCH -J Task2
#SBATCH --partition=instruction
#SBATCH -o task2.out -e task2.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge


g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec -o task2_nosimd

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec -DUSE_SIMD -o task2_simd

./task2_nosimd 1000000 4
./task2_simd 1000000 4
