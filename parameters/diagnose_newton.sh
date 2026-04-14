#!/bin/bash
# Diagnose sgd_newton_diag: HP boundaries, metric dynamics, effective LR.
# Usage: bash parameters/diagnose_newton.sh [task] [iteration] [results_dir]

TASK="${1:-mnist_mlp}"
ITR="${2:-7}"
RESULTS="${3:-results}"
DIR="$RESULTS/$TASK/sgd_newton_diag/itr_$ITR"

echo "==========================================="
echo "Newton diagonal diagnosis: $TASK itr_$ITR"
echo "==========================================="

python3 -c "
import csv, json, glob, os, statistics

d = '$DIR'
files = sorted(glob.glob(os.path.join(d, '*.csv')))
n = len(files)
print(f'Total trials: {n}')
if n == 0:
    raise SystemExit()

# HP sweep bounds (from optimizer_registry.py _PARAM_BOUNDS)
bounds = {
    'learning_rate': (1e-6, 1e1, 'log'),
    'momentum': (0.0, 0.99, 'lin'),
    'beta_s': (0.01, 0.5, 'log'),
    'weight_decay': (1e-6, 1e-1, 'log'),
    'metric_clip': (1.0, 5.0, 'lin'),
}

def _f(row, key):
    v = row.get(key, '')
    try: return float(v)
    except: return None

# Parse all trials
records = []
for f in files:
    with open(f) as fh:
        meta = json.loads(fh.readline().lstrip('# '))
        rows = list(csv.DictReader(fh))
    s = meta.get('summary', {})
    cfg = meta.get('config', {})

    # Grab diagnostic columns at last epoch
    last = rows[-1] if rows else {}
    # Grab diagnostic columns at epoch 10 (early training)
    early = {}
    for row in rows:
        try:
            if int(float(row.get('epoch', -1))) == 10:
                early = row
                break
        except: pass

    records.append({
        'file': os.path.basename(f),
        'acc': s.get('final_max_val_acc', 0),
        'config': cfg,
        # Metric diagnostics (last epoch)
        'eff_lr_mean': _f(last, 'diag/eff_lr/mean'),
        'eff_lr_max': _f(last, 'diag/eff_lr/max'),
        'eff_lr_min': _f(last, 'diag/eff_lr/min'),
        'metric_condition': _f(last, 'diag/metric_condition'),
        'clipped_frac': _f(last, 'diag/clipped_frac'),
        'log_diag_mean': _f(last, 'diag/log_diag/mean'),
        'log_diag_std': _f(last, 'diag/log_diag/std'),
        # Early diagnostics (epoch 10)
        'eff_lr_mean_e10': _f(early, 'diag/eff_lr/mean'),
        'clipped_frac_e10': _f(early, 'diag/clipped_frac'),
        'metric_condition_e10': _f(early, 'diag/metric_condition'),
    })

# Sort by accuracy
records.sort(key=lambda r: r['acc'], reverse=True)
best = records[:50]  # top 50
worst = records[-50:]  # bottom 50

# =====================================================================
# 1. HP VALUES FOR BEST TRIALS — check for boundary clustering
# =====================================================================
print()
print('=' * 70)
print('1. HP VALUES: best 50 trials (check for boundary clustering)')
print('=' * 70)
print()
print(f'{\"HP\":<16s} {\"sweep_range\":>15s}  {\"best50_min\":>10s} {\"best50_med\":>10s} {\"best50_max\":>10s}  {\"at_lo\":>5s} {\"at_hi\":>5s}')
print('-' * 85)

for hp, (lo, hi, scale) in bounds.items():
    vals = [r['config'].get(hp) for r in best if r['config'].get(hp) is not None]
    if not vals:
        continue
    vals_f = [float(v) for v in vals]
    vals_f.sort()
    med = vals_f[len(vals_f)//2]

    # Check boundary clustering (within 10% of boundary in log/lin space)
    if scale == 'log':
        import math
        log_range = math.log10(hi) - math.log10(lo)
        at_lo = sum(1 for v in vals_f if math.log10(v) - math.log10(lo) < 0.1 * log_range)
        at_hi = sum(1 for v in vals_f if math.log10(hi) - math.log10(v) < 0.1 * log_range)
    else:
        lin_range = hi - lo
        at_lo = sum(1 for v in vals_f if v - lo < 0.1 * lin_range)
        at_hi = sum(1 for v in vals_f if hi - v < 0.1 * lin_range)

    range_str = f'[{lo}, {hi}]'
    print(f'{hp:<16s} {range_str:>15s}  {vals_f[0]:>10.6f} {med:>10.6f} {vals_f[-1]:>10.6f}  {at_lo:>5d} {at_hi:>5d}')

# =====================================================================
# 2. HP CORRELATION WITH ACCURACY
# =====================================================================
print()
print('=' * 70)
print('2. HP VALUES: best 10 vs worst 10 trials')
print('=' * 70)
print()

print('Best 10:')
for r in records[:10]:
    c = r['config']
    print(f'  {r[\"acc\"]:.4f}  lr={c.get(\"learning_rate\",0):.4e}  mom={c.get(\"momentum\",0):.3f}  '
          f'beta_s={c.get(\"beta_s\",0):.4f}  clip={c.get(\"metric_clip\",0):.2f}  '
          f'wd={c.get(\"weight_decay\",0):.4e}')

print()
print('Worst 10:')
for r in records[-10:]:
    c = r['config']
    print(f'  {r[\"acc\"]:.4f}  lr={c.get(\"learning_rate\",0):.4e}  mom={c.get(\"momentum\",0):.3f}  '
          f'beta_s={c.get(\"beta_s\",0):.4f}  clip={c.get(\"metric_clip\",0):.2f}  '
          f'wd={c.get(\"weight_decay\",0):.4e}')

# =====================================================================
# 3. METRIC DYNAMICS: effective LR, condition number, clipping
# =====================================================================
print()
print('=' * 70)
print('3. METRIC DYNAMICS (last epoch): best 50 vs all trials')
print('=' * 70)
print()

def _stats(vals, name):
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f'  {name}: no data')
        return
    vals.sort()
    n = len(vals)
    print(f'  {name} (n={n}): min={vals[0]:.4g}  med={vals[n//2]:.4g}  max={vals[-1]:.4g}')

print('Best 50 trials:')
_stats([r['eff_lr_mean'] for r in best], 'eff_lr/mean')
_stats([r['eff_lr_max'] for r in best], 'eff_lr/max')
_stats([r['eff_lr_min'] for r in best], 'eff_lr/min')
_stats([r['metric_condition'] for r in best], 'metric_condition')
_stats([r['clipped_frac'] for r in best], 'clipped_frac')

print()
print('All trials:')
_stats([r['eff_lr_mean'] for r in records], 'eff_lr/mean')
_stats([r['eff_lr_max'] for r in records], 'eff_lr/max')
_stats([r['metric_condition'] for r in records], 'metric_condition')
_stats([r['clipped_frac'] for r in records], 'clipped_frac')

# =====================================================================
# 4. EARLY TRAINING (epoch 10): is the metric already saturated?
# =====================================================================
print()
print('=' * 70)
print('4. EARLY TRAINING (epoch 10): best 50 trials')
print('=' * 70)
print()

_stats([r['eff_lr_mean_e10'] for r in best], 'eff_lr/mean @ep10')
_stats([r['clipped_frac_e10'] for r in best], 'clipped_frac @ep10')
_stats([r['metric_condition_e10'] for r in best], 'metric_condition @ep10')

# =====================================================================
# 5. BEST TRIAL DETAILED TRAJECTORY
# =====================================================================
print()
print('=' * 70)
print('5. BEST TRIAL TRAJECTORY: metric evolution over training')
print('=' * 70)
print()

best_file = os.path.join(d, records[0]['file'])
with open(best_file) as fh:
    fh.readline()
    rows = list(csv.DictReader(fh))

print(f'Trial: {records[0][\"file\"]} (acc={records[0][\"acc\"]:.4f})')
c = records[0]['config']
print(f'Config: lr={c.get(\"learning_rate\",0):.4e}  mom={c.get(\"momentum\",0):.3f}  '
      f'beta_s={c.get(\"beta_s\",0):.4f}  clip={c.get(\"metric_clip\",0):.2f}')
print()

fmt = '{:>5s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>8s}  {:>8s}'
print(fmt.format('epoch', 'loss', 'acc', 'eff_lr_m', 'eff_lr_max', 'cond', 'clip_f', 'rho'))
print('-' * 90)

for row in rows:
    ep = row.get('epoch', '')
    try:
        ep_int = int(float(ep))
    except:
        continue
    if ep_int not in (0, 1, 2, 5, 10, 20, 50, 100, 150, 199):
        continue
    loss = row.get('train_loss', '')
    acc = row.get('test_acc', '')
    eff_m = row.get('diag/eff_lr/mean', '')
    eff_max = row.get('diag/eff_lr/max', '')
    cond = row.get('diag/metric_condition', '')
    clip = row.get('diag/clipped_frac', '')
    rho = row.get('diag/h_diag/rho_global', '')
    print(fmt.format(ep, loss[:10] if loss else '', acc[:10] if acc else '',
                     eff_m[:10] if eff_m else '', eff_max[:10] if eff_max else '',
                     cond[:10] if cond else '', clip[:8] if clip else '',
                     rho[:8] if rho else ''))
" 2>&1 | head -200
