#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task3
#SBATCH --partition=instruction
#SBATCH -o task3.out -e task3.err
./task3
