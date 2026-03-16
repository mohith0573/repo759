#!/usr/bin/env zsh
#SBATCH -c 2
#SBATCH -J Task1_thrust
#SBATCH --partition=instruction
#SBATCH -o task1_thrust.out -e task1_thrust.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1

module purge
module load nvidia/cuda/13.0

# compile
nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_thrust

# run example
./task1_thrust 1024
