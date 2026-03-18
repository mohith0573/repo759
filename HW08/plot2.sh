##!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Plot2
#SBATCH --partition=instruction
#SBATCH -o plot2.out -e plot2.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge
source ~/myenv/bin/activate
python3 plot2.py
deactivate
