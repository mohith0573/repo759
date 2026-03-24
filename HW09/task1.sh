#!/usr/bin/env zsh
#SBATCH -J Task1
#SBATCH --partition=instruction
#SBATCH -o task1.out -e task1.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge


g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

./task1 5040000 4
