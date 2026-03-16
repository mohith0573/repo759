#!/usr/bin/env zsh
#SBATCH -c 4
#SBATCH -J Task3
#SBATCH --partition=instruction
#SBATCH -o task3.out -e task3.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=4

module purge

# compile
g++ task3.cpp -Wall -O3 -std=c++17 -fopenmp -o task3

# run
./task3
