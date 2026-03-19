#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J Plot3_t
#SBATCH --partition=instruction
#SBATCH -o plot3_t.out -e plot3_t.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge
source ~/myenv/bin/activate
python3 plot3_t.py
deactivate
