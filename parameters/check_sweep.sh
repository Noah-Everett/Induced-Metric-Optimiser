#!/bin/bash
# Quick monitoring script for a running sweep.
# Usage: bash parameters/check_sweep.sh [results_dir]
# Default: results/mnist_mlp/sgd_newton_diag/itr_7

DIR="${1:-results/mnist_mlp/sgd_newton_diag/itr_7}"

echo "==========================================="
echo "Sweep monitor: $DIR"
echo "==========================================="

# 1. Job status
echo ""
echo "--- SLURM jobs ---"
squeue -u "$USER" -n imo-mnist 2>/dev/null || echo "(not on SLURM)"

# 2-5. All Python analysis in one block
echo ""
python3 -c "
import csv, json, glob, os

d = '$DIR'
files = sorted(glob.glob(os.path.join(d, '*.csv')))
n = len(files)
print(f'--- Progress: {n} trials completed ---')
if n == 0:
    print('No results yet.')
    raise SystemExit()

# Parse all files
records = []
for f in files:
    with open(f) as fh:
        meta = json.loads(fh.readline().lstrip('# '))
        rows = list(csv.DictReader(fh))
    s = meta.get('summary', {})
    last = rows[-1] if rows else {}
    records.append({
        'file': os.path.basename(f),
        'time': s.get('total_training_time_sec', 0),
        'acc': s.get('final_max_val_acc', 0),
        'rho': last.get('diag/h_diag/rho_global', 'N/A'),
        'neg': last.get('diag/h_diag/neg_frac', 'N/A'),
        'pruned': s.get('pruned', False),
    })

# Timing
times = [r['time'] for r in records if r['time'] > 0]
if times:
    times.sort()
    med = times[len(times)//2]
    print(f'')
    print(f'--- Wall-clock time (seconds/trial) ---')
    print(f'  median={med:.1f}s  min={times[0]:.1f}s  max={times[-1]:.1f}s  n={len(times)}')

# Accuracy
accs = sorted([r['acc'] for r in records], reverse=True)
print(f'')
print(f'--- Accuracy ---')
print(f'  best={accs[0]:.4f}  median={accs[len(accs)//2]:.4f}  worst={accs[-1]:.4f}')

# Diagnostics (first 5 files)
print(f'')
print(f'--- Diagnostics (last epoch, first 5 trials) ---')
for r in records[:5]:
    print(f'  {r[\"file\"]}: rho={r[\"rho\"]}, neg_frac={r[\"neg\"]}')

# Pruning
pruned = sum(1 for r in records if r['pruned'])
if pruned:
    print(f'')
    print(f'--- Pruning: {pruned}/{n} trials pruned ---')
"

# Latest SLURM log
echo ""
echo "--- Last 10 lines of latest SLURM log ---"
latest=$(ls -t slurm_logs/*.out 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    tail -10 "$latest"
else
    echo "(no SLURM logs found)"
fi
