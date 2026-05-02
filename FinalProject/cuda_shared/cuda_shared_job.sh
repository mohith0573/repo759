#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J CUDAShared
#SBATCH --partition=instruction
#SBATCH -o cuda_shared.out
#SBATCH -e cuda_shared.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module purge
module load nvidia/cuda/13.0.0

cd "$SLURM_SUBMIT_DIR"

H=64
W=64
Cin=16
Cout=8
K=3
REPEATS=20
WRITE_MATRICES=0

INPUT_FILE="input.csv"
KERNEL_FILE="kernel.csv"
RESULT_FILE="cuda_shared_results.csv"

if [[ ! -f "$INPUT_FILE" || ! -f "$KERNEL_FILE" ]]; then
    echo "ERROR: input.csv and kernel.csv must exist in the cuda_shared directory." >&2
    echo "Copy them from your sequential directory first, for example:" >&2
    echo "  cp ../seq/input.csv ./input.csv" >&2
    echo "  cp ../seq/kernel.csv ./kernel.csv" >&2
    exit 1
fi

rm -f conv_cuda_shared cuda_shared_results.csv cuda_shared_filter_*.csv

nvcc main_cuda_shared.cu conv_cuda_shared.cu \
    -O3 \
    -std=c++17 \
    -Xcompiler -Wall \
    -Xptxas -O3 \
    -o conv_cuda_shared

./conv_cuda_shared $H $W $Cin $Cout $K $REPEATS "$INPUT_FILE" "$KERNEL_FILE" $WRITE_MATRICES > "$RESULT_FILE"

echo "CUDA shared-memory job completed."
echo "Results written to $RESULT_FILE"
