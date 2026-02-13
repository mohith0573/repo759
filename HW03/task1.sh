#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1
#SBATCH --partition=instruction
#SBATCH -o task1.out -e task1.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0
./task1
