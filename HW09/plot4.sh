#!/usr/bin/env zsh
#SBATCH -c 10
#SBATCH -J Plot4
#SBATCH --partition=instruction
#SBATCH -o plot4.out -e plot4.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge
source ~/myenv/bin/activate
python3 plot4.py
deactivate
