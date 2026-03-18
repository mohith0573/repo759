#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Task3
#SBATCH --partition=instruction
#SBATCH -o task3.out -e task3.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -fopenmp -o task3

# run (n=1e6, t=8, ts=1024 assumed good)
./task3 1000000 8 1024
