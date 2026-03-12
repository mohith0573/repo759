#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_plot
#SBATCH --partition=instruction
#SBATCH -o plot1.out -e plot1.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
module purge
source ~/myenv/bin/activate
python3 plot1.py
deactivate
