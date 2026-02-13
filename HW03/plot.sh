#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task3_plot
#SBATCH --partition=instruction
#SBATCH -o plot.out -e plot.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module purge
source ~/myenv/bin/activate
python3 task3_plot.py
deactivate
