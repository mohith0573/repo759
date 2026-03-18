#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Task2
#SBATCH --partition=instruction
#SBATCH -o task2.out -e task2.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -fopenmp -o task2

# run (n=1024, threads=8 for example)
./task2 1024 8
