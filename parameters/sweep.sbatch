#!/bin/bash
#SBATCH --job-name=imo-sweep
#SBATCH --account=schwartz_lab
#SBATCH --array=0-25
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-0:00:00
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# SLURM Batch Script for Optimizer Sweeps
# =============================================================================
# Array size = num_optimizers * max(num_functions, 1)
#   small_examples (5 functions): --array=0-129  (26 * 5)
#   single-task sweeps:           --array=0-25   (26 * 1)
#
# Usage:
#   sbatch slurm_sweep.sh                    # Run all jobs
#   sbatch --array=0-4 slurm_sweep.sh        # Run first optimizer only (5 functions)
#   sbatch --array=0,5,10 slurm_sweep.sh     # Run specific jobs
#
# To see optimizer index mapping, run:
#   grep -n "OPTIMIZERS\[" slurm_sweep.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration (edit these as needed)
# -----------------------------------------------------------------------------

# Task settings
TASK="parameters/sweep_shake.py"           # Sweep script to run
RESULTS_DIR="results"                      # Results directory
ITERATION=1                                # Batch iteration number

# Functions to sweep (leave empty for single-task sweeps like mnist_mlp)
FUNCTIONS=()

# Sweep settings
NUM_RUNS=200                               # Number of runs per optimizer
BACKEND="local"                            # "local" or "wandb"
SEARCH="random"                            # Search method: "random", "bayes" (TPE), "grid" (QMC)
PRUNER="none"                              # Pruner: "none", "hyperband", "median", "percentile"
SEED=42                                    # Random seed for reproducibility

# Training settings (empty = use task default)
BATCH_SIZE=""                              # Override batch size
VAL_FREQ=1                                 # Validation frequency (epochs)

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
module load python
mamba activate induced-metric-optimiser

# JAX/CUDA configuration
export XLA_FLAGS='--xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_PREALLOCATE='true'
export XLA_PYTHON_CLIENT_ALLOCATOR='platform'

# -----------------------------------------------------------------------------
# Optimizer array (indexed 0-25)
# -----------------------------------------------------------------------------
OPTIMIZERS=(
    "adam"                      # 0
    "adamw"                     # 1
    "sgd"                       # 2
    "muon"                      # 3
    "sgd_metric"                # 4
    "sgd_log_metric"            # 5
    "sgd_rms"                   # 6
    "sgd_learn_scalar"          # 7
    "sgd_learn_scalar_log"      # 8
    "sgd_learn_diag"            # 9
    "sgd_learn_diag_log"        # 10
    "sgd_offdiag_0_l"           # 11
    "sgd_offdiag_0_m"           # 12
    "sgd_offdiag_0_theta"       # 13
    "sgd_offdiag_l_0"           # 14
    "sgd_offdiag_l_l"           # 15
    "sgd_offdiag_l_m"           # 16
    "sgd_offdiag_l_theta"       # 17
    "sgd_offdiag_m_0"           # 18
    "sgd_offdiag_m_l"           # 19
    "sgd_offdiag_m_m"           # 20
    "sgd_offdiag_m_theta"       # 21
    "sgd_offdiag_theta_0"       # 22
    "sgd_offdiag_theta_l"       # 23
    "sgd_offdiag_theta_m"       # 24
    "sgd_offdiag_theta_theta"   # 25
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
echo "Pruner: $PRUNER"
echo "Seed: $SEED"
echo "Batch size: ${BATCH_SIZE:-default}"
echo "Val freq: $VAL_FREQ"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# Change to submission directory (where sbatch was invoked)
cd "$SLURM_SUBMIT_DIR"

# Build command with all options
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
