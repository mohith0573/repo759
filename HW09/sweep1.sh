#!/usr/bin/env zsh
#SBATCH -J Sweep1
#SBATCH --partition=instruction
#SBATCH -o sweep1.out -e sweep1.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

# clear file
> task1_data.txt

n=5040000

for t in {1..10}
do
  ./task1 $n $t >> task1_data.txt
done
