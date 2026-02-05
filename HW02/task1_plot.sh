#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_Plot
#SBATCH --partition=instruction
#SBATCH -o task1_plot.out -e task1_plot.err
#SBATCH --mem=16G
module load gcc/14.3.0
source ~/myenv/bin/activate
for i in {10..30}
do
    n=$((2**i))
    ./task1 $n >> task1_times.txt
done
python task1_plot.py
deactivate
