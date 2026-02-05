#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task2
#SBATCH --partition=instruction
#SBATCH -o task2.out -e task2.err
./task2 4 3
