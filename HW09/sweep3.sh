#!/usr/bin/env zsh

#SBATCH -J Sweep3
#SBATCH --partition=instruction
#SBATCH -o sweep3.out -e sweep3.err
#SBATCH --time=00:10:00

#SBATCH --nodes=1 --ntasks-per-node=2

module purge
module load mpi/mpich/4.0.2

mpicxx task3.cpp -O3 -o task3

> task3_data.txt

for i in {1..25}
do
  n=$((2**i))
  srun -n 2 ./task3 $n >> task3_data.txt
done
