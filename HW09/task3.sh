#!/usr/bin/env zsh

#SBATCH -J Task3
#SBATCH --partition=instruction
#SBATCH -o task3.out -e task3.err
#SBATCH --time=00:05:00

#SBATCH --nodes=1 --ntasks-per-node=2

module purge
module load mpi/mpich/4.0.2

mpicxx task3.cpp -Wall -O3 -o task3

srun -n 2 ./task3 1024
