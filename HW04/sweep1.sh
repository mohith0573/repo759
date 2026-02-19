#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_sweep
#SBATCH --partition=instruction
#SBATCH -o sweep1.out -e sweep1.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module load nvidia/cuda/13.0.0

> task1_times_1024.txt
> task1_times_256.txt

for ((i=5;i<=14;i++)); do
    n=$((2**i))

    t1=$(./task1 $n 1024 | tail -n 1)
    echo $t1 >> task1_times_1024.txt

    t2=$(./task1 $n 256 | tail -n 1)
    echo $t2 >> task1_times_256.txt
done
