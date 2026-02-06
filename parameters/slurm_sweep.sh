#!/bin/bash
#SBATCH --job-name=imo-sweep
#SBATCH --account=schwartz_lab
#SBATCH --array=0-25
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=$HOME/slurm_logs/%A_%a.out
#SBATCH --error=$HOME/slurm_logs/%A_%a.err

# =============================================================================
# SLURM Batch Script for Optimizer Sweeps
# =============================================================================
# Usage:
#   sbatch slurm_sweep.sh                    # Run all 26 optimizers
#   sbatch --array=0-3 slurm_sweep.sh        # Run first 4 optimizers only
#   sbatch --array=0,5,10 slurm_sweep.sh     # Run specific optimizers
#
# To see optimizer index mapping, run:
#   grep -n "OPTIMIZERS\[" slurm_sweep.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration (edit these as needed)
# -----------------------------------------------------------------------------

# Task settings
TASK="parameters/sweep_mnist_mlp.py"   # Sweep script to run
RESULTS_DIR="results"                  # Results directory
ITERATION=2                            # Batch iteration number

# Sweep settings
NUM_RUNS=1000                          # Number of runs per optimizer
BACKEND="local"                        # "local" or "wandb"
SEARCH="random"                        # Search method: "random", "bayes" (TPE), "grid" (QMC)
PRUNER="none"                          # Pruner: "none", "hyperband", "median", "percentile"
SEED=42                                # Random seed for reproducibility

# Training settings (empty = use task default)
BATCH_SIZE=""                          # Override batch size
VAL_FREQ=1                             # Validation frequency (epochs)

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
# Optimizer array (indexed 0-26)
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

# Get optimizer for this array task
OPTIMIZER="${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$OPTIMIZER" ]; then
    echo "Error: Invalid array task ID $SLURM_ARRAY_TASK_ID"
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
echo "Optimizer: $OPTIMIZER"
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

cd /Users/noah-everett/Documents/Research/Induced-Metric-Optimiser/parameters

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
