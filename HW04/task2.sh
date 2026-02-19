#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task2
#SBATCH --partition=instruction
#SBATCH -o task2.out -e task2.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
module purge
module load nvidia/cuda/13.0.0
./task2 1024 128 1024
