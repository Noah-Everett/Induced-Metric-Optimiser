#!/bin/bash
#SBATCH --job-name=imo-adam-imo-probe
#SBATCH --account=schwartz_lab
#SBATCH --array=0-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# IMO-166: Adam-IMO bolt-on probe (one-off; not a permanent task).
# =============================================================================
# 6 array tasks, one per (optimiser, xi/clip_norm) combination:
#   0: adam       (no extra args)
#   1: adamw      (--weight_decay 1e-4)
#   2: adam_clip  (--clip_norm 1.0)
#   3: adam_imo   (--xi 0.01)
#   4: adam_imo   (--xi 0.1)
#   5: adam_imo   (--xi 1.0)
#
# Each task runs the explicit 5-LR x 3-seed grid, 30 epochs each, on CIFAR-10
# SmallCNN (~90K params) with cosine LR + standard augmentation.  Output lands
# at results/cifar10_simple_cnn/<optimiser>/itr_0/run_*.csv via SweepLogger,
# matching the on-disk format every other sweep produces.
#
# Usage from repo root:
#     sbatch notebooks/adam_imo_bolt_on.sbatch
# =============================================================================

TASK="notebooks/adam_imo_bolt_on.py"
RESULTS_DIR="results"
ITERATION=0

module load python
mamba activate induced-metric-optimiser

export XLA_FLAGS='--xla_gpu_autotune_level=4'
export XLA_PYTHON_CLIENT_PREALLOCATE='true'
export JAX_DEFAULT_MATMUL_PRECISION='tensorfloat32'

mkdir -p slurm_logs "$RESULTS_DIR"

case "$SLURM_ARRAY_TASK_ID" in
    0) OPT="adam";       EXTRA_ARGS="" ;;
    1) OPT="adamw";      EXTRA_ARGS="--weight_decay 1e-4" ;;
    2) OPT="adam_clip";  EXTRA_ARGS="--clip_norm 1.0" ;;
    3) OPT="adam_imo";   EXTRA_ARGS="--xi 0.01" ;;
    4) OPT="adam_imo";   EXTRA_ARGS="--xi 0.1" ;;
    5) OPT="adam_imo";   EXTRA_ARGS="--xi 1.0" ;;
    *)
        echo "Error: invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected 0-5)"
        exit 1
        ;;
esac

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "Optimiser: $OPT  Extra: $EXTRA_ARGS"
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

cd "$SLURM_SUBMIT_DIR"

CMD="python $TASK \
    --optimiser $OPT \
    --iteration $ITERATION \
    --results_dir $RESULTS_DIR \
    --diagnostics \
    $EXTRA_ARGS"

echo "Running: $CMD"
eval $CMD
EXIT=$?

echo "==========================================="
echo "Exit code: $EXIT"
echo "==========================================="
exit $EXIT
