#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1
#SBATCH --partition=instruction
#SBATCH -o task1.out -e task1.err
./task1 6
