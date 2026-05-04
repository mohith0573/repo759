#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J PlotConv
#SBATCH --partition=instruction
#SBATCH -o plots.out
#SBATCH -e plots.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module purge

cd "$SLURM_SUBMIT_DIR"

source ~/myenv/bin/activate
python3 generate_all_plots.py
deactivate
