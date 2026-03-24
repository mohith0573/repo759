#!/usr/bin/env zsh
#SBATCH -c 10
#SBATCH -J Plot3
#SBATCH --partition=instruction
#SBATCH -o plot3.out -e plot3.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge
source ~/myenv/bin/activate
python3 plot3.py
deactivate
