#!/bin/bash
# Batch runner for optimizer sweeps
# Runs 5 trials each for priority optimizers

set -e
cd "$(dirname "$0")"

RESULTS_DIR="../results"
NUM_RUNS=5

# Priority order (most interesting first)
OPTIMIZERS=(
    "adam"
    "adamw"
    "sgd"
    "muon"
    "sgd_metric"
    "sgd_log_metric"
    "sgd_rms"
    "sgd_learn_scalar"
    "sgd_learn_scalar_log"
    "sgd_learn_diag"
    "sgd_learn_diag_log"
    "sgd_offdiag_0_l"
    "sgd_offdiag_0_m"
    "sgd_offdiag_0_theta"
    "sgd_offdiag_l_0"
    "sgd_offdiag_l_l"
    "sgd_offdiag_l_m"
    "sgd_offdiag_l_theta"
    "sgd_offdiag_m_0"
    "sgd_offdiag_m_l"
    "sgd_offdiag_m_m"
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

    # Check which run indices already exist
    runs_to_do=()
    for i in $(seq 0 $((NUM_RUNS - 1))); do
        result_dir="$RESULTS_DIR/mnist_mlp/$opt/run_${i}"
        # Check if directory exists and has at least one JSON file
        if [ ! -d "$result_dir" ] || [ -z "$(find "$result_dir" -name "*.json" -type f 2>/dev/null)" ]; then
            runs_to_do+=($i)
        fi
    done

    # Skip if all runs are complete
    if [ ${#runs_to_do[@]} -eq 0 ]; then
        echo "  All $NUM_RUNS runs already complete, skipping"
        continue
    fi

    echo "  Running ${#runs_to_do[@]} missing runs: ${runs_to_do[*]}"

    # Run each missing index
    for run_idx in "${runs_to_do[@]}"; do
        python3 sweep_mnist_mlp.py --optimiser "$opt" --num_runs 1 --index "$run_idx" --backend local --results_dir "$RESULTS_DIR" 2>&1 || {
            echo "  FAILED: $opt run $run_idx (exit code $?)"
            continue
        }
    done

    echo "$(date '+%H:%M:%S') Completed: $opt"
    echo ""
done

echo "=========================================="
echo "$(date '+%H:%M:%S') All sweeps completed!"
echo "=========================================="
