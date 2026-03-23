"""Generate a 4-panel figure showing progressive basin filling on Himmelblau."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'parameters'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'optimisers'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

jax.config.update('jax_enable_x64', True)

from sweep_small_examples import TEST_FUNCTIONS
from basin_filling.metric_core import (
    MetricBasinFillingConfig, _single_run, _compute_metric_fields,
)
from basin_filling.core import Basin
from basin_filling.strategies import detection, radius

# ---------------------------------------------------------------------------
# Run basin filling, capturing per-run archives
# ---------------------------------------------------------------------------
f = TEST_FUNCTIONS['himmelblau']
loss_fn, grad_fn = f['func'], f['grad']
initial = f['initial']

cfg = MetricBasinFillingConfig(
    detect_stuck=detection.loss_plateau(threshold=1e-7),
    estimate_radius=radius.fixed(value=3.0),
    lr=0.1, xi=0.1, beta_m=0.9,
    steering_strength=2.0,
    detection_patience=50, max_runs=8, max_steps_per_run=5000,
)

archive = []
run_data = []  # (trajectory, basin, archive_snapshot_before)

for run_idx in range(cfg.max_runs):
    archive_before = list(archive)
    traj, basin = _single_run(loss_fn, grad_fn, initial, archive, run_idx, cfg)
    if basin is not None:
        archive.append(basin)
        run_data.append((traj, basin, archive_before))
    else:
        break

# Deduplicate: pick first 4 runs that find distinct minima
seen = set()
panels = []
for traj, basin, arch_before in run_data:
    key = (round(basin.center[0], 0), round(basin.center[1], 0))
    if key not in seen:
        seen.add(key)
        panels.append((traj, basin, arch_before))
    if len(panels) == 4:
        break

# Pad if fewer than 4
while len(panels) < 4:
    panels.append(panels[-1])

# ---------------------------------------------------------------------------
# Known minima
# ---------------------------------------------------------------------------
true_minima = [
    (3.0, 2.0), (3.5844, -1.8481), (-2.8051, 3.1313), (-3.7793, -3.2832)
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 9.5), constrained_layout=True)
axes = axes.flatten()

# Contour grid
x = np.linspace(-5.5, 5.5, 400)
y = np.linspace(-5.5, 5.5, 400)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

for idx, ax in enumerate(axes):
    traj, basin, arch_before = panels[idx]

    # Contours
    ax.contourf(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap='bone_r',
                norm=LogNorm(vmin=0.1, vmax=1000), alpha=0.6)
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 15), colors='gray',
               linewidths=0.3, norm=LogNorm(vmin=0.1, vmax=1000))

    # Filled basins — show Phi(theta) damping as heatmap
    if arch_before:
        Phi = np.ones_like(Z)
        for b in arch_before:
            dist_sq = (X - b.center[0])**2 + (Y - b.center[1])**2
            psi_k = np.exp(-dist_sq / (2.0 * b.radius**2))
            Phi *= (1.0 - psi_k)
        # Show 1-Phi: high where damping is removed (near filled basins)
        ax.contourf(X, Y, 1.0 - Phi,
                    levels=[0.05, 0.2, 0.5, 0.8, 0.95],
                    cmap='Blues', alpha=0.35, zorder=2)
        ax.contour(X, Y, 1.0 - Phi,
                   levels=[0.5], colors='steelblue',
                   linewidths=1.0, linestyles='--', zorder=2)
        for b in arch_before:
            ax.plot(b.center[0], b.center[1], 'x', color='steelblue',
                    markersize=7, markeredgewidth=1.5, zorder=5)

    # Trajectory
    pts = np.array([s.params for s in traj])
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#d62728',
            linewidth=0.8, alpha=0.7, zorder=3)

    # Start and end markers
    ax.plot(pts[0, 0], pts[0, 1], 'o', color='white', markersize=6,
            markeredgecolor='black', markeredgewidth=1.2, zorder=6)
    ax.plot(pts[-1, 0], pts[-1, 1], '*', color='#d62728', markersize=12,
            markeredgecolor='black', markeredgewidth=0.5, zorder=6)

    # Newly found basin — solid circle
    # True minima
    for mx, my in true_minima:
        ax.plot(mx, my, '+', color='black', markersize=8,
                markeredgewidth=1.5, zorder=7)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    n_filled = len(arch_before)
    ax.set_title(f'Run {idx+1}:  found ({basin.center[0]:.1f}, {basin.center[1]:.1f})'
                 f'    [{n_filled} basins filled]',
                 fontsize=10)
    ax.set_xlabel(r'$\theta_1$', fontsize=9)
    ax.set_ylabel(r'$\theta_2$', fontsize=9)
    ax.tick_params(labelsize=8)

fig.suptitle(
    'Basin filling via spatially varying metric on Himmelblau\n'
    r'$\xi_0\!=\!0.1,\ \alpha_0\!=\!2,\ r\!=\!3$'
    r'$\quad\mathbf{+}$ true minima'
    r'$\quad\bigstar$ found'
    r'$\quad\bigcirc$ start',
    fontsize=10,
)

outpath = os.path.join(os.path.dirname(__file__), 'basin_filling_himmelblau.pdf')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
outpath_png = outpath.replace('.pdf', '.png')
fig.savefig(outpath_png, dpi=200, bbox_inches='tight')
print(f'Saved: {outpath}')
print(f'Saved: {outpath_png}')
plt.close()
