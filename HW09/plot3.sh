#!/usr/bin/env zsh

#SBATCH -J Plot3
#SBATCH --partition=instruction
#SBATCH -o plot3.out -e plot3.err
#SBATCH --time=00:05:00

#SBATCH --nodes=1 --ntasks-per-node=2

module purge
source ~/myenv/bin/activate
python3 plot3.py
deactivate
