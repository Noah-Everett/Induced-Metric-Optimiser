#!/bin/bash
#SBATCH --job-name=imo-consolidate
#SBATCH --account=schwartz_lab
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=slurm_logs/%j_consolidate.out
#SBATCH --error=slurm_logs/%j_consolidate.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=END,FAIL

module load python
mamba activate induced-metric-optimiser

mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

python parameters/consolidate_results.py \
    --results_dir results \
    --skip_existing \
    --workers 8
