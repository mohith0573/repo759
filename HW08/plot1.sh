##!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Plot1
#SBATCH --partition=instruction
#SBATCH -o plot1.out -e plot1.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge
source ~/myenv/bin/activate
python3 plot1.py
deactivate
