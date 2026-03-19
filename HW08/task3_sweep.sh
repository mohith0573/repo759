#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Task3_sweep
#SBATCH --partition=instruction
#SBATCH -o task3_sweep.out -e task3_sweep.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

> times_task3_ts.txt

# ts = 2^1 → 2^10
for ((i=1;i<=10;i++)); do
    ts=$((2**i))
    ./task3 1000000 8 $ts | tail -n 1 >> times_task3_ts.txt
done
