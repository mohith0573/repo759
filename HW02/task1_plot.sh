#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_Plot
#SBATCH --partition=instruction
#SBATCH -o task1_plot.out -e task1_plot.err
module load gcc
module load Python/3.11.4-GCCcore-12.2.0
for i in {10..30}
do
    n=$((2**i))
    ./task1 $n >> task1_times.txt
done
python plot_task1.py
