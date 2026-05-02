#!/usr/bin/env zsh
#SBATCH -c 1
#SBATCH -J PyRefConv
#SBATCH --partition=instruction
#SBATCH -o python_ref.out
#SBATCH -e python_ref.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

module purge

cd "$SLURM_SUBMIT_DIR"

H=128
W=128
Cin=16
Cout=8
K=3
REPEATS=1
WRITE_MATRICES=0

INPUT_FILE="input.csv"
KERNEL_FILE="kernel.csv"
RESULT_FILE="python_reference_results.csv"

if [[ ! -f "$INPUT_FILE" || ! -f "$KERNEL_FILE" ]]; then
    echo "ERROR: input.csv and kernel.csv must exist in the python directory." >&2
    echo "Copy them from seq/:" >&2
    echo "  cp ../seq/input.csv ./input.csv" >&2
    echo "  cp ../seq/kernel.csv ./kernel.csv" >&2
    exit 1
fi



source ~/myenv/bin/activate
python3 reference.py \
    --H $H --W $W --Cin $Cin --Cout $Cout --K $K \
    --repeats $REPEATS \
    --input "$INPUT_FILE" \
    --kernel "$KERNEL_FILE" \
    --write-matrices $WRITE_MATRICES \
    --prefix python_reference \
    > "$RESULT_FILE"

echo "Python reference job completed. Results written to $RESULT_FILE"
deactivate
