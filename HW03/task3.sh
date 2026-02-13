#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task3
#SBATCH --partition=instruction
#SBATCH -o task3.out -e task3.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0
./task3 1024
