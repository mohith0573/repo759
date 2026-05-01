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

# Run from the directory where this script is located: results_comparison/
cd "${0:A:h}"
source ~/myenv/bin/activate

python3 compare_all_with_python_reference.py \
    --H 64 \
    --W 64 \
    --Cin 3 \
    --Cout 8 \
    --K 3 \
    --tol 1e-4 \
    > comparison_report.txt

echo "Comparison completed."
echo "Read comparison_report.txt and comparison_summary.csv."
deactivate
