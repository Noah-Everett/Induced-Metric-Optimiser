#!/bin/bash
# Compare GN diagonal (itr_7) vs Hutchinson (itr_6) for sgd_newton_diag.
# Usage: bash parameters/compare_gn_vs_hutch.sh

HUTCH="results/mnist_mlp/sgd_newton_diag/itr_6"
GN="results/mnist_mlp/sgd_newton_diag/itr_7"

echo "==========================================="
echo "GN diagonal (itr_7) vs Hutchinson (itr_6)"
echo "==========================================="

python3 -c "
import csv, json, glob, os, statistics

def parse_dir(d):
    records = []
    for f in sorted(glob.glob(os.path.join(d, '*.csv'))):
        with open(f) as fh:
            meta = json.loads(fh.readline().lstrip('# '))
            rows = list(csv.DictReader(fh))
        s = meta.get('summary', {})
        # Grab diagnostics at epoch 10, 50, 100, and last
        diag_at = {}
        for row in rows:
            ep = int(float(row.get('epoch', -1)))
            if ep in (10, 50, 100, 199):
                diag_at[ep] = {
                    'rho': row.get('diag/h_diag/rho_global', ''),
                    'neg_frac': row.get('diag/h_diag/neg_frac', ''),
                    's_gap_mean': row.get('diag/s_target_gap/mean', ''),
                    's_gap_max': row.get('diag/s_target_gap/max', ''),
                    'loss': row.get('train_loss', ''),
                    'acc': row.get('test_acc', ''),
                }
        records.append({
            'file': os.path.basename(f),
            'time': s.get('total_training_time_sec', 0),
            'acc': s.get('final_max_val_acc', 0),
            'pruned': s.get('pruned', False),
            'diag_at': diag_at,
        })
    return records

hutch = parse_dir('$HUTCH')
gn = parse_dir('$GN')

print(f'Hutchinson (itr_6): {len(hutch)} trials')
print(f'GN diagonal (itr_7): {len(gn)} trials')

# --- 1. Accuracy distribution ---
print()
print('--- Accuracy distribution ---')
for label, recs in [('Hutch', hutch), ('GN   ', gn)]:
    if not recs:
        print(f'  {label}: no data')
        continue
    accs = sorted([r['acc'] for r in recs], reverse=True)
    n = len(accs)
    p = [0, 5, 25, 50, 75, 95, 100]
    vals = [accs[min(int(q/100*n), n-1)] for q in [100-q for q in p]]
    # top-1, top-5%, median
    print(f'  {label} (n={n:4d}): best={accs[0]:.4f}  p95={accs[int(0.05*n)]:.4f}  median={accs[n//2]:.4f}  p5={accs[int(0.95*n)]:.4f}')

# --- 2. Timing ---
print()
print('--- Timing (seconds/trial) ---')
for label, recs in [('Hutch', hutch), ('GN   ', gn)]:
    times = sorted([r['time'] for r in recs if r['time'] > 0])
    if times:
        print(f'  {label}: median={times[len(times)//2]:.1f}s  min={times[0]:.1f}s  max={times[-1]:.1f}s')

# --- 3. Diagnostics at key epochs (best trial from each) ---
print()
print('--- Diagnostics for best trial (by accuracy) ---')
for label, recs in [('Hutch', hutch), ('GN   ', gn)]:
    if not recs:
        continue
    best = max(recs, key=lambda r: r['acc'])
    print(f'  {label} best: {best[\"file\"]} (acc={best[\"acc\"]:.4f})')
    for ep in [10, 50, 100, 199]:
        d = best['diag_at'].get(ep, {})
        rho = d.get('rho', 'N/A')
        neg = d.get('neg_frac', 'N/A')
        sgap = d.get('s_gap_mean', 'N/A')
        loss = d.get('loss', 'N/A')
        acc = d.get('acc', 'N/A')
        print(f'    ep={ep:3d}: rho={str(rho):>8s}  neg_frac={str(neg):>5s}  s_gap={str(sgap):>8s}  loss={str(loss):>8s}  acc={str(acc):>7s}')

# --- 4. Median diagnostics across all trials at epoch 100 ---
print()
print('--- Median diagnostics at epoch 100 (all trials) ---')
for label, recs in [('Hutch', hutch), ('GN   ', gn)]:
    rhos, negs, sgaps = [], [], []
    for r in recs:
        d = r['diag_at'].get(100, {})
        try: rhos.append(float(d.get('rho', '')))
        except: pass
        try: negs.append(float(d.get('neg_frac', '')))
        except: pass
        try: sgaps.append(float(d.get('s_gap_mean', '')))
        except: pass
    if rhos:
        print(f'  {label}: rho={statistics.median(rhos):.2f}  neg_frac={statistics.median(negs):.4f}  s_gap={statistics.median(sgaps):.4f}  (n={len(rhos)})')
    else:
        print(f'  {label}: no diagnostic data')

# --- 5. Top-10 accuracy comparison ---
print()
print('--- Top 10 trials by accuracy ---')
for label, recs in [('Hutch', hutch), ('GN   ', gn)]:
    top = sorted(recs, key=lambda r: r['acc'], reverse=True)[:10]
    accs_str = ', '.join(f'{r[\"acc\"]:.4f}' for r in top)
    print(f'  {label}: {accs_str}')
"
