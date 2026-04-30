#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J OMPConv
#SBATCH --partition=instruction
#SBATCH -o openmp.out -e openmp.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

cd "$SLURM_SUBMIT_DIR"

H=64
W=64
Cin=3
Cout=8
K=3
REPEATS=5
THREADS=20
WRITE_MATRICES=1

INPUT_FILE="input.csv"
KERNEL_FILE="kernel.csv"
RESULT_FILE="openmp_results.csv"

if [[ ! -f "$INPUT_FILE" || ! -f "$KERNEL_FILE" ]]; then
    echo "ERROR: input.csv and kernel.csv must exist inside the openmp directory." >&2
    echo "Copy them from your seq directory first, for example:" >&2
    echo "  cp ../seq/input.csv ./input.csv" >&2
    echo "  cp ../seq/kernel.csv ./kernel.csv" >&2
    exit 1
fi

g++ main_openmp.cpp conv_openmp.cpp -Wall -O3 -std=c++17 -o conv_openmp -fopenmp

export OMP_NUM_THREADS=$THREADS

./conv_openmp $H $W $Cin $Cout $K $REPEATS "$INPUT_FILE" "$KERNEL_FILE" $WRITE_MATRICES $THREADS > "$RESULT_FILE"

echo "OpenMP job completed. Results written to $RESULT_FILE"
echo "Output matrices written as openmp_filter_*.csv"
