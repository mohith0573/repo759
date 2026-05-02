#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J CUDANaiveConv
#SBATCH --partition=instruction
#SBATCH -o cuda_naive.out -e cuda_naive.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=1

module purge
module load nvidia/cuda/13.0.0

cd "$SLURM_SUBMIT_DIR"

H=512
W=512
Cin=16
Cout=8
K=3
REPEATS=20
WRITE_MATRICES=0

INPUT_FILE="input.csv"
KERNEL_FILE="kernel.csv"
RESULT_FILE="cuda_naive_results.csv"

if [[ ! -f "$INPUT_FILE" || ! -f "$KERNEL_FILE" ]]; then
    echo "ERROR: input.csv and kernel.csv must exist inside cuda_naive/." >&2
    echo "Copy them from sequential first, for example:" >&2
    echo "  cp ../seq/input.csv ./input.csv" >&2
    echo "  cp ../seq/kernel.csv ./kernel.csv" >&2
    exit 1
fi

nvcc main_cuda_naive.cu conv_cuda_naive.cu -O3 -std=c++17 -Xcompiler -Wall -Xptxas -O3 -o conv_cuda_naive

./conv_cuda_naive $H $W $Cin $Cout $K $REPEATS "$INPUT_FILE" "$KERNEL_FILE" $WRITE_MATRICES > "$RESULT_FILE"

echo "CUDA naive job completed. Results written to $RESULT_FILE"
echo "Output matrices written as cuda_naive_filter_*.csv"
