#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Task2_sweep
#SBATCH --partition=instruction
#SBATCH -o task2_sweep.out -e task2_sweep.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

> times_task2_hw8.txt

for ((t=1;t<=20;t++)); do
    ./task2 1024 $t | tail -n 1 >> times_task2_hw8.txt
done
