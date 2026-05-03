#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J CompareConv
#SBATCH --partition=instruction
#SBATCH -o comparison.out
#SBATCH -e comparison.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module purge

cd "$SLURM_SUBMIT_DIR"

python3 compare_all_with_python_reference.py \
    --H 128 \
    --W 128 \
    --Cin 16 \
    --Cout 8 \
    --K 3 \
    --tol 1e-4 \
    > comparison_report.txt

echo "Comparison completed."
echo "Read comparison_report.txt, comparison_summary.csv, and input_kernel_consistency.csv."
