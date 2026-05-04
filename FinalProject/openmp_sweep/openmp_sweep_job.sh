#!/usr/bin/env zsh
#SBATCH -c 20
#SBATCH -J OMPSweep
#SBATCH --partition=instruction
#SBATCH -o openmp_sweep.out
#SBATCH -e openmp_sweep.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20

module purge

# This script is meant to be run from:
#   FinalProject/openmp_sweep/
#
# Submit using:
#   sbatch openmp_sweep_job.sh

cd "$SLURM_SUBMIT_DIR"

# -------------------------------------------------------------------
# Project structure assumed:
#
# FinalProject/
# ├── inputs/
# ├── openmp/
# ├── exe_results/
# └── openmp_sweep/
# -------------------------------------------------------------------

PROJECT_DIR=".."
INPUTS_DIR="$PROJECT_DIR/inputs"
OPENMP_DIR="$PROJECT_DIR/openmp"
EXE_RESULTS_DIR="$PROJECT_DIR/exe_results"

mkdir -p "$EXE_RESULTS_DIR"

# -------------------------------------------------------------------
# Benchmark configuration
# Your current benchmark setup:
#   image sizes = 64, 128, 256, 512
#   Cin = 16
#   Cout = 8
#   K = 3
# -------------------------------------------------------------------

SIZES=(64 128 256 512)
Cin=16
Cout=8
K=3
REPEATS=5
WRITE_MATRICES=0

# Euler instruction partition usually gives 20 CPUs in your scripts.
THREAD_LIST=(1 2 4 8 16 20)

COMBINED_OUT="$EXE_RESULTS_DIR/openmp_sweep_all.csv"

echo "method,H,W,Cin,Cout,K,threads,repeats,time_ms,total_time_ms,checksum,input_file,kernel_file" > "$COMBINED_OUT"

# Compile OpenMP executable once.
cd "$OPENMP_DIR"

rm -f conv_openmp

g++ main_openmp.cpp conv_openmp.cpp \
    -Wall \
    -O3 \
    -std=c++17 \
    -o conv_openmp \
    -fopenmp

if [[ $? -ne 0 ]]; then
    echo "ERROR: OpenMP compilation failed." >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

for SIZE in "${SIZES[@]}"; do

    H=$SIZE
    W=$SIZE

    # Supports both input_64.csv and input_64 naming styles.
    INPUT_SRC="$INPUTS_DIR/input_${SIZE}.csv"
    KERNEL_SRC="$INPUTS_DIR/kernel_${SIZE}.csv"

    if [[ ! -f "$INPUT_SRC" ]]; then
        INPUT_SRC="$INPUTS_DIR/input_${SIZE}"
    fi

    if [[ ! -f "$KERNEL_SRC" ]]; then
        KERNEL_SRC="$INPUTS_DIR/kernel_${SIZE}"
    fi

    if [[ ! -f "$INPUT_SRC" || ! -f "$KERNEL_SRC" ]]; then
        echo "ERROR: Could not find input/kernel files for SIZE=$SIZE" >&2
        echo "Expected one of:" >&2
        echo "  $INPUTS_DIR/input_${SIZE}.csv or $INPUTS_DIR/input_${SIZE}" >&2
        echo "  $INPUTS_DIR/kernel_${SIZE}.csv or $INPUTS_DIR/kernel_${SIZE}" >&2
        exit 1
    fi

    echo "Running OpenMP sweep for ${SIZE}x${SIZE}..."

    cp "$INPUT_SRC" "$OPENMP_DIR/input.csv"
    cp "$KERNEL_SRC" "$OPENMP_DIR/kernel.csv"

    SIZE_OUT="$EXE_RESULTS_DIR/openmp_sweep_${SIZE}.csv"
    echo "method,H,W,Cin,Cout,K,threads,repeats,time_ms,total_time_ms,checksum,input_file,kernel_file" > "$SIZE_OUT"

    cd "$OPENMP_DIR"

    for T in "${THREAD_LIST[@]}"; do
        export OMP_NUM_THREADS=$T
        export OMP_PROC_BIND=close
        export OMP_PLACES=cores

        ./conv_openmp $H $W $Cin $Cout $K $REPEATS input.csv kernel.csv $WRITE_MATRICES $T > tmp_openmp_sweep_result.csv

        # Append only data row, not header.
        awk 'NR==2 {print}' tmp_openmp_sweep_result.csv >> "$SIZE_OUT"
        awk 'NR==2 {print}' tmp_openmp_sweep_result.csv >> "$COMBINED_OUT"

        echo "  completed SIZE=$SIZE threads=$T"
    done

    rm -f tmp_openmp_sweep_result.csv

    cd "$SLURM_SUBMIT_DIR"
done

echo "OpenMP sweep completed."
echo "Per-size CSV files written to $EXE_RESULTS_DIR/openmp_sweep_<size>.csv"
echo "Combined CSV written to $COMBINED_OUT"
