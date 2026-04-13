#!/bin/bash
#SBATCH --job-name=imo-sweep
#SBATCH --account=schwartz_lab
#SBATCH --array=0-7
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# IMO-96: Saddle Escape Benchmark
# =============================================================================
# 4 optimizers × 2 functions = 8 array jobs (--array=0-7)
#
# Usage:
#   sbatch parameters/sweep_saddle_bench.sbatch
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TASK="parameters/sweep_saddle.py"
RESULTS_DIR="results"
ITERATION=2

FUNCTIONS=("saddle" "weighted_saddle")

NUM_RUNS=1000
BACKEND="local"
SEARCH="bayes"
PRUNER="none"
SEED=42

BATCH_SIZE=""
VAL_FREQ=1

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
module load python
mamba activate induced-metric-optimiser

export JAX_PLATFORMS=cpu

# -----------------------------------------------------------------------------
# Optimizer array (indexed 0-3)
# -----------------------------------------------------------------------------
OPTIMIZERS=(
    "adam"                      # 0
    "sgd"                      # 1
    "sgd_learn_diag_curv"      # 2
    "sgd_newton_diag"          # 3
)

# -----------------------------------------------------------------------------
# 2D array indexing: (optimizer, function)
# -----------------------------------------------------------------------------
NUM_FUNCTIONS=${#FUNCTIONS[@]}
if [ "$NUM_FUNCTIONS" -eq 0 ]; then
    NUM_FUNCTIONS=1
fi

OPTIMIZER_IDX=$((SLURM_ARRAY_TASK_ID / NUM_FUNCTIONS))
FUNCTION_IDX=$((SLURM_ARRAY_TASK_ID % NUM_FUNCTIONS))

OPTIMIZER="${OPTIMIZERS[$OPTIMIZER_IDX]}"

if [ ${#FUNCTIONS[@]} -gt 0 ]; then
    FUNCTION="${FUNCTIONS[$FUNCTION_IDX]}"
fi

if [ -z "$OPTIMIZER" ]; then
    echo "Error: Invalid array task ID $SLURM_ARRAY_TASK_ID (optimizer index $OPTIMIZER_IDX out of range)"
    exit 1
fi

# -----------------------------------------------------------------------------
# Create log directory if needed
# -----------------------------------------------------------------------------
mkdir -p slurm_logs

# -----------------------------------------------------------------------------
# Run the sweep
# -----------------------------------------------------------------------------
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Optimizer [$OPTIMIZER_IDX]: $OPTIMIZER"
echo "Function [$FUNCTION_IDX]: ${FUNCTION:-N/A}"
echo "Task: $TASK"
echo "Num runs: $NUM_RUNS"
echo "Iteration: $ITERATION"
echo "Search: $SEARCH"
echo "Node: $(hostname)"
echo "==========================================="

cd "$SLURM_SUBMIT_DIR"

CMD="python $TASK"
CMD="$CMD --optimiser $OPTIMIZER"
CMD="$CMD --num_runs $NUM_RUNS"
CMD="$CMD --backend $BACKEND"
CMD="$CMD --iteration $ITERATION"
CMD="$CMD --results_dir $RESULTS_DIR"
CMD="$CMD --search $SEARCH"
CMD="$CMD --pruner $PRUNER"
CMD="$CMD --seed $SEED"
CMD="$CMD --val_freq $VAL_FREQ"
CMD="$CMD --diagnostics"

if [ -n "$FUNCTION" ]; then
    CMD="$CMD --function $FUNCTION"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?

echo "=========================================="
echo "Completed with exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
