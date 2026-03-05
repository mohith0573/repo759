#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep1.out -e sweep1.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0

# compile
nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std=c++17 -o task1

# clear output file
> times_task1.txt

# sweep n = 2^5 ... 2^11
for ((i=5;i<=11;i++)); do
    n=$((2**i))
    ./task1 $n 20 >> times_task1.txt
done
