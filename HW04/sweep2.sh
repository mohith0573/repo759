#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task2_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep2.out -e sweep2.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module load nvidia/cuda/13.0.0

R=128

> task2_times_1024.txt
> task2_times_256.txt

for ((i=10;i<=29;i++)); do
    n=$((2**i))

    t1=$(./task2 $n $R 1024 | tail -n 1)
    echo $t1 >> task2_times_1024.txt

    t2=$(./task2 $n $R 256 | tail -n 1)
    echo $t2 >> task2_times_256.txt
done
