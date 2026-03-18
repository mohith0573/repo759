#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Task3_sweep_t
#SBATCH --partition=instruction
#SBATCH -o task3_sweep_t.out -e task3_sweep_t.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -fopenmp -o task3

> times_task3_t.txt

for ((t=1;t<=20;t++)); do
    ./task3 1000000 $t 1024 | tail -n 1 >> times_task3_t.txt
done
