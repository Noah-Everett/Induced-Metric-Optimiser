#!/bin/bash
# Compare all optimizers for a given task and iteration.
# Usage: bash parameters/compare_optimizers.sh [task] [iteration] [results_dir]
# Default: mnist_mlp, itr_7, results

TASK="${1:-mnist_mlp}"
ITR="${2:-7}"
RESULTS="${3:-results}"
BASE="$RESULTS/$TASK"

echo "==========================================="
echo "Optimizer comparison: $TASK  itr_$ITR"
echo "==========================================="

python3 -c "
import csv, json, glob, os, statistics

base = '$BASE'
itr = 'itr_$ITR'

# Find all optimizers with results
optimizers = sorted([
    d for d in os.listdir(base)
    if os.path.isdir(os.path.join(base, d, itr))
]) if os.path.isdir(base) else []

if not optimizers:
    print(f'No results found in {base}/*/itr_{itr}/')
    raise SystemExit()

data = {}
for opt in optimizers:
    d = os.path.join(base, opt, itr)
    files = sorted(glob.glob(os.path.join(d, '*.csv')))
    records = []
    for f in files:
        with open(f) as fh:
            meta = json.loads(fh.readline().lstrip('# '))
            rows = list(csv.DictReader(fh))
        s = meta.get('summary', {})
        last = rows[-1] if rows else {}
        records.append({
            'time': s.get('total_training_time_sec', 0),
            'acc': s.get('final_max_val_acc', 0),
        })
    data[opt] = records

# --- Summary table ---
print()
fmt = '{:<25s} {:>5s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}'
print(fmt.format('optimizer', 'n', 'best', 'p95', 'median', 'p5', 'time_s', 'speedup'))
print('-' * 95)

# Get baseline time (adam or first optimizer)
baseline_time = None
for label, recs in sorted(data.items()):
    times = sorted([r['time'] for r in recs if r['time'] > 0])
    if times and label == 'adam':
        baseline_time = times[len(times)//2]
if baseline_time is None:
    for label, recs in sorted(data.items()):
        times = sorted([r['time'] for r in recs if r['time'] > 0])
        if times:
            baseline_time = times[len(times)//2]
            break

for label, recs in sorted(data.items()):
    n = len(recs)
    if n == 0:
        print(f'{label:<25s} {0:>5d}')
        continue
    accs = sorted([r['acc'] for r in recs], reverse=True)
    times = sorted([r['time'] for r in recs if r['time'] > 0])
    med_time = times[len(times)//2] if times else 0

    best = f'{accs[0]:.4f}'
    p95 = f'{accs[max(int(0.05*n),0)]:.4f}' if n > 1 else best
    med = f'{accs[n//2]:.4f}'
    p5 = f'{accs[min(int(0.95*n), n-1)]:.4f}' if n > 1 else med
    t = f'{med_time:.1f}'
    sp = f'{med_time/baseline_time:.2f}x' if baseline_time and med_time else ''

    print(fmt.format(label, str(n), best, p95, med, p5, t, sp))

# --- Top 5 per optimizer ---
print()
print('--- Top 5 trials per optimizer ---')
for label, recs in sorted(data.items()):
    top = sorted(recs, key=lambda r: r['acc'], reverse=True)[:5]
    accs_str = ', '.join(f'{r[\"acc\"]:.4f}' for r in top)
    times_str = ', '.join(f'{r[\"time\"]:.0f}s' for r in top)
    print(f'  {label}:')
    print(f'    acc:  {accs_str}')
    print(f'    time: {times_str}')

# --- Cost-adjusted comparison (accuracy per GPU-hour) ---
print()
print('--- Efficiency: trials per GPU-hour at median speed ---')
for label, recs in sorted(data.items()):
    times = sorted([r['time'] for r in recs if r['time'] > 0])
    if times:
        med_time = times[len(times)//2]
        trials_per_hr = 3600.0 / med_time
        print(f'  {label}: {trials_per_hr:.0f} trials/hr ({med_time:.1f}s each)')
"
