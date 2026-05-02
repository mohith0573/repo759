#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J GenData
#SBATCH --partition=instruction
#SBATCH -o generate_data.out -e generate_data.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=1

module purge

# Run this job from inside the seq directory.
cd "$SLURM_SUBMIT_DIR"

H=256
W=256
Cin=16
Cout=8
K=3
SEED=759

source ~/myenv/bin/activate
python3 generate_data.py --H $H --W $W --Cin $Cin --Cout $Cout --K $K --seed $SEED --input input.csv --kernel kernel.csv
deactivate
