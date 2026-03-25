#!/usr/bin/env zsh
#SBATCH -J Task4
#SBATCH --partition=instruction
#SBATCH -o task4.out -e task4.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge
source ~/myenv/bin/activate
python3 plot4.py
deactivate
