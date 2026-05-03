#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J PlotExeResults
#SBATCH --partition=instruction
#SBATCH -o plot_results.out
#SBATCH -e plot_results.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module purge
cd "$SLURM_SUBMIT_DIR"
source ~/myenv/bin/activate
python3 plot_from_exe_results.py --exe-dir ../exe_results --sizes 64 128 256 512 > plot_results_report.txt
deactivate
