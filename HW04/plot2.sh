#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task2_plot
#SBATCH --partition=instruction
#SBATCH -o plot2.out -e plot2.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module purge
source ~/myenv/bin/activate
python3 plot2.py
deactivate
