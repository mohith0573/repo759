#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J SeqConv
#SBATCH --partition=instruction
#SBATCH -o task_seq.out -e task_seq.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=1

module purge

# Run this job from inside the seq directory.
cd "$SLURM_SUBMIT_DIR"

H=512
W=512
Cin=16
Cout=8
K=3
REPEATS=5
WRITE_MATRICES=0

INPUT_FILE="input.csv"
KERNEL_FILE="kernel.csv"

if [[ ! -f "$INPUT_FILE" || ! -f "$KERNEL_FILE" ]]; then
    echo "ERROR: input.csv and kernel.csv must exist in this seq directory." >&2
    echo "Generate them first using:" >&2
    echo "  sbatch generate_data_job.sh" >&2
    echo "or:" >&2
    echo "  python3 generate_data.py --H $H --W $W --Cin $Cin --Cout $Cout --K $K --input input.csv --kernel kernel.csv" >&2
    exit 1
fi

g++ main_sequential.cpp conv_sequential.cpp -Wall -O3 -std=c++17 -o conv_sequential

./conv_sequential $H $W $Cin $Cout $K $REPEATS "$INPUT_FILE" "$KERNEL_FILE" $WRITE_MATRICES > sequential_results.csv

echo "Sequential run completed."
echo "Timing file: sequential_results.csv"
echo "Output matrices: sequential_filter_0.csv ... sequential_filter_$((Cout - 1)).csv"
