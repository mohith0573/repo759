#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task6
#SBATCH -o task6.out -e task6.err
./task6 6
