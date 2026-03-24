#!/usr/bin/env zsh
#SBATCH -J Sweep2
#SBATCH --partition=instruction
#SBATCH -o sweep2.out -e sweep2.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=10

module purge
g++ task2.cpp montecarlo.cpp -O3 -fopenmp -fno-tree-vectorize -march=native -o task2_nosimd
g++ task2.cpp montecarlo.cpp -O3 -fopenmp -fno-tree-vectorize -march=native -DUSE_SIMD -o task2_simd

> nosimd.txt
> simd.txt

n=1000000

for t in {1..10}
do
  ./task2_nosimd $n $t >> nosimd.txt
  ./task2_simd $n $t >> simd.txt
done
