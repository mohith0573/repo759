#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_thrust_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep_task1_thrust.out -e sweep_task1_thrust.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1

module purge
module load nvidia/cuda/13.0

# compile
nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_thrust

# clear output
> times_thrust.txt

# sweep n = 2^10 ... 2^20
for ((i=10;i<=20;i++)); do
    n=$((2**i))
    ./task1_thrust $n | tail -1 >> times_thrust.txt
done
