#!/bin/bash
#SBATCH --job-name=imo-cifar
#SBATCH --account=schwartz_lab
#SBATCH --array=0-3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-0:00:00
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# IMO-98: CIFAR-10 ResNet-18 Benchmark
# =============================================================================
# 4 optimizers × 1 task = 4 array jobs (--array=0-3)
# Each job runs NUM_RUNS Bayesian HP trials of one optimizer.
#
# Usage:
#   sbatch parameters/sweep_cifar.sbatch
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TASK="parameters/sweep_cifar10_resnet18.py"
RESULTS_DIR="results"
ITERATION=3

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

# JAX/CUDA configuration
export XLA_FLAGS='--xla_gpu_autotune_level=4'
export XLA_PYTHON_CLIENT_PREALLOCATE='true'
export JAX_DEFAULT_MATMUL_PRECISION='tensorfloat32'

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
# Array indexing (single-task, no function dimension)
# -----------------------------------------------------------------------------
OPTIMIZER_IDX=$SLURM_ARRAY_TASK_ID
OPTIMIZER="${OPTIMIZERS[$OPTIMIZER_IDX]}"

if [ -z "$OPTIMIZER" ]; then
    echo "Error: Invalid array task ID $SLURM_ARRAY_TASK_ID (out of range)"
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
echo "Task: $TASK"
echo "Num runs: $NUM_RUNS"
echo "Iteration: $ITERATION"
echo "Search: $SEARCH"
echo "Pruner: $PRUNER"
echo "Seed: $SEED"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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
