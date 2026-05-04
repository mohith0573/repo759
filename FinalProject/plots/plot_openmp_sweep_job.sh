#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J PlotOMPSweep
#SBATCH --partition=instruction
#SBATCH -o plot_openmp_sweep.out
#SBATCH -e plot_openmp_sweep.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module purge

cd "$SLURM_SUBMIT_DIR"
source ~/myenv/bin/activate
python3 plot_openmp_sweep.py
deactivate