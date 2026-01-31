#!/bin/bash
# Batch runner for off-diagonal optimizer sweeps
# Runs 5 trials each for priority optimizers

set -e
cd "$(dirname "$0")"

RESULTS_DIR="../results"
NUM_RUNS=5

# Priority order (most interesting first)
OPTIMIZERS=(
    "sgd_offdiag_m_l"
    "sgd_offdiag_l_l"
    "sgd_offdiag_m_m"
    "sgd_offdiag_0_0"
    "sgd_offdiag_0_l"
    "sgd_offdiag_0_m"
    "sgd_offdiag_0_theta"
    "sgd_offdiag_l_0"
    "sgd_offdiag_l_theta"
    "sgd_offdiag_m_0"
    "sgd_offdiag_m_theta"
    "sgd_offdiag_theta_0"
    "sgd_offdiag_theta_l"
    "sgd_offdiag_theta_m"
    "sgd_offdiag_theta_theta"
)

for opt in "${OPTIMIZERS[@]}"; do
    echo "=========================================="
    echo "$(date '+%H:%M:%S') Starting: $opt"
    echo "=========================================="
    
    # Check if already has enough results
    existing=$(find "$RESULTS_DIR/mnist_mlp/$opt/" -type f -name "*.json" 2>/dev/null | wc -l)
    if [ "$existing" -ge "$NUM_RUNS" ]; then
        echo "  Already has $existing results, skipping"
        continue
    fi
    
    python3 sweep_mnist_mlp.py --optimiser "$opt" --num_runs "$NUM_RUNS" --backend local --results_dir "$RESULTS_DIR" 2>&1 || {
        echo "  FAILED: $opt (exit code $?)"
        continue
    }
    
    echo "$(date '+%H:%M:%S') Completed: $opt"
    echo ""
done

echo "=========================================="
echo "$(date '+%H:%M:%S') All sweeps completed!"
echo "=========================================="
