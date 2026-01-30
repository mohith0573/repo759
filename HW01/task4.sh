#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J FirstSlurm
#SBATCH --partition=instruction
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
hostname
