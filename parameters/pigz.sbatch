#!/bin/bash
#
# Fast parallel compression with tar + pigz on FASRC
#
#SBATCH --job-name pigz
#SBATCH --partition sapphire                # or 'intermediate' if you need 14-day limit
#SBATCH --account=schwartz_lab
#SBATCH -N 1
#SBATCH --cpus-per-task=112                 # number of threads for pigz
#SBATCH --mem=64G                           # use all RAM on the node; or set e.g. 64G
#SBATCH --time 1-00:00:00                   # 1 day; adjust as needed
#SBATCH --output pigz_%j.out
#SBATCH --error pigz_%j.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

####### USER SETTINGS #######

# Directory you want to compress (can be large, many files)
SRC_DIR="/n/home03/neverett/Induced-Metric-Optimiser/results"

# Where you want the final archive to live (e.g. lab storage, $HOME, etc.)
FINAL_DEST="/n/home03/neverett/Induced-Metric-Optimiser"

# Compression level (1 = fastest, 9 = smallest; 1 is good for speed)
PIGZ_LEVEL=1

##############################

# Basic checks
if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: SRC_DIR does not exist or is not a directory: $SRC_DIR"
  exit 1
fi

if [ ! -d "$FINAL_DEST" ]; then
  echo "ERROR: FINAL_DEST does not exist or is not a directory: $FINAL_DEST"
  exit 1
fi

# Load pigz if it's a module on FASRC (safe even if pigz is already in PATH)
module load pigz 2>/dev/null || true

echo "Job ID:      $SLURM_JOB_ID"
echo "Partition:   $SLURM_JOB_PARTITION"
echo "Cores:       $SLURM_CPUS_PER_TASK"
echo "Source dir:  $SRC_DIR"
echo "Final dest:  $FINAL_DEST"

# Use node-local scratch (fast I/O)
SCRATCH_DIR="/n/netscratch/schwartz_lab/Everyone/neverett/$SLURM_JOB_ID"
mkdir -p "$SCRATCH_DIR"

echo "Scratch dir: $SCRATCH_DIR"
echo "Copying data to scratch..."
rsync -aH --info=progress2 "$SRC_DIR" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

BASENAME="$(basename "$SRC_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="${BASENAME}_${TIMESTAMP}.tar.gz"

echo "Creating archive: $ARCHIVE_NAME"
echo "Using pigz with $SLURM_CPUS_PER_TASK threads at level -$PIGZ_LEVEL"

tar -I "pigz -p $SLURM_CPUS_PER_TASK -$PIGZ_LEVEL" \
    -cf "$ARCHIVE_NAME" "$BASENAME"

echo "Archive size on scratch:"
ls -lh "$ARCHIVE_NAME"

echo "Moving archive to final destination..."
mv "$ARCHIVE_NAME" "$FINAL_DEST/"

echo "Cleaning up scratch copy..."
rm -rf "$SCRATCH_DIR/$BASENAME"

echo "Done."
echo "Final archive: $FINAL_DEST/$ARCHIVE_NAME"