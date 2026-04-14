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
        # Parse per-epoch data for convergence analysis
        epochs = []
        for row in rows:
            ep = {}
            try: ep['epoch'] = int(float(row.get('epoch', -1)))
            except: continue
            for k in ('test_acc', 'train_loss', 'test_mse', 'train_mse'):
                if k in row and row[k]:
                    try: ep[k] = float(row[k])
                    except: pass
            epochs.append(ep)
        records.append({
            'time': s.get('total_training_time_sec', 0),
            'acc': s.get('final_max_val_acc', 0),
            'loss': s.get('final_min_val_loss', None),
            'epochs': epochs,
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

# --- Convergence speed: epochs to reach target accuracy ---
print()
print('--- Convergence speed (median epochs to target, best 100 trials) ---')

# Detect metric: use test_acc if available, else test_mse
has_acc = any(
    any('test_acc' in ep for ep in r['epochs'])
    for recs in data.values() for r in recs
)

if has_acc:
    targets = [0.90, 0.95, 0.97]
    metric_key = 'test_acc'
    metric_name = 'accuracy'
    higher_better = True
else:
    # Regression: use test_mse, lower is better
    # Pick targets based on observed range
    all_losses = [r['loss'] for recs in data.values() for r in recs
                  if r['loss'] is not None]
    if all_losses:
        best_loss = min(all_losses)
        targets = [best_loss * 10, best_loss * 3, best_loss * 1.5]
        targets = [round(t, 4) for t in targets]
    else:
        targets = []
    metric_key = 'test_mse'
    metric_name = 'test_mse'
    higher_better = False

if targets:
    header = '{:<25s}'.format('optimizer')
    for t in targets:
        if has_acc:
            header += f'  {\"ep@\"+str(t):>10s}'
        else:
            header += f'  {\"ep@\"+str(t):>12s}'
    print(f'  metric: {metric_name} ({\">=\" if higher_better else \"<=\"} target)')
    print(f'  using best 100 trials by final metric')
    print()
    print(f'  {header}')
    print(f'  ' + '-' * len(header))

    for label, recs in sorted(data.items()):
        # Take best 100 trials
        if higher_better:
            top = sorted(recs, key=lambda r: r['acc'], reverse=True)[:100]
        else:
            top = sorted(recs, key=lambda r: r['loss'] if r['loss'] is not None else 1e10)[:100]

        row = f'{label:<25s}'
        for target in targets:
            epochs_to_target = []
            for r in top:
                found = None
                for ep in r['epochs']:
                    val = ep.get(metric_key)
                    if val is None:
                        continue
                    if (higher_better and val >= target) or (not higher_better and val <= target):
                        found = ep['epoch']
                        break
                if found is not None:
                    epochs_to_target.append(found)

            if epochs_to_target:
                med = sorted(epochs_to_target)[len(epochs_to_target)//2]
                frac = len(epochs_to_target)
                row += f'  {med:>4d} ({frac:>3d})'
            else:
                row += f'  {\"never\":>10s}'
        print(f'  {row}')

    print()
    print(f'  Format: median_epoch (num_trials_reaching_target out of top 100)')

# --- HP robustness: fraction reaching thresholds ---
print()
print('--- HP robustness (fraction of ALL trials reaching target) ---')

if has_acc:
    thresholds = [0.90, 0.95, 0.97]
    header = '{:<25s} {:>5s}'.format('optimizer', 'n')
    for t in thresholds:
        header += f'  {\">=\" + str(t):>8s}'
    print(f'  {header}')
    print(f'  ' + '-' * len(header))

    for label, recs in sorted(data.items()):
        n = len(recs)
        row = f'{label:<25s} {n:>5d}'
        for t in thresholds:
            count = sum(1 for r in recs if r['acc'] >= t)
            pct = count / n * 100 if n > 0 else 0
            row += f'  {pct:>7.1f}%'
        print(f'  {row}')
else:
    if targets:
        header = '{:<25s} {:>5s}'.format('optimizer', 'n')
        for t in targets:
            header += f'  {\"<=\" + str(t):>10s}'
        print(f'  {header}')
        print(f'  ' + '-' * len(header))

        for label, recs in sorted(data.items()):
            n = len(recs)
            row = f'{label:<25s} {n:>5d}'
            for t in targets:
                count = sum(1 for r in recs if r['loss'] is not None and r['loss'] <= t)
                pct = count / n * 100 if n > 0 else 0
                row += f'  {pct:>9.1f}%'
            print(f'  {row}')

# --- Efficiency: trials per GPU-hour ---
print()
print('--- Efficiency: trials per GPU-hour at median speed ---')
for label, recs in sorted(data.items()):
    times = sorted([r['time'] for r in recs if r['time'] > 0])
    if times:
        med_time = times[len(times)//2]
        trials_per_hr = 3600.0 / med_time
        print(f'  {label}: {trials_per_hr:.0f} trials/hr ({med_time:.1f}s each)')
"
