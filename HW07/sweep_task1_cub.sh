#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_cub_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep_task1_cub.out -e sweep_task1_cub.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1

module purge
module load nvidia/cuda/13.0

# compile
nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_cub

# clear output
> times_cub.txt

# sweep n = 2^10 ... 2^20
for ((i=10;i<=20;i++)); do
    n=$((2**i))
    ./task1_cub $n | tail -1 >> times_cub.txt
done
