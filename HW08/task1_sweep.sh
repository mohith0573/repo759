#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Task1_sweep
#SBATCH --partition=instruction
#SBATCH -o task1_sweep.out -e task1_sweep.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge
g++ task1.cpp matmul.cpp -O3 -std=c++17 -fopenmp -o task1

> times_task1.txt

for t in {1..20}; do
    ./task1 1024 $t | tail -n 1 >> times_task1.txt
done
