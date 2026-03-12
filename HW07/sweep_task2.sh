#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task2_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep_task2.out -e sweep_task2.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1

module purge
module load nvidia/cuda/13.0

# compile
nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# clear output
> times_task2.txt

# sweep n = 2^5 ... 2^20
for ((i=5;i<=20;i++)); do
    n=$((2**i))
    ./task2 $n | tail -1 >> times_task2.txt
done
