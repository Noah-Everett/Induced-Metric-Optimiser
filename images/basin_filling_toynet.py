"""Basin filling on a 2-parameter toy neural network.

Model: y = tanh(w1 * x) + tanh(w2 * x)  (2 hidden units, fixed output weights)
Data:  x in [-3, 3], y = tanh(1.5x) + tanh(0.5x) + noise
Loss:  MSE over 200 data points

The permutation symmetry (w1, w2) <-> (w2, w1) and the sign symmetry of
tanh create multiple equivalent minima.  Basin filling discovers them.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'parameters'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'optimisers'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

jax.config.update('jax_enable_x64', True)

from basin_filling.core import (
    Basin, BasinFillingConfig, run_basin_filling,
)
from basin_filling.strategies import detection, radius, walkthrough, containment
import optax

# ---------------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------------
np.random.seed(42)
N_DATA = 200
x_data = np.linspace(-3, 3, N_DATA)
y_data = np.tanh(1.5 * x_data) + np.tanh(0.5 * x_data) + 0.05 * np.random.randn(N_DATA)

x_jax = jnp.array(x_data)
y_jax = jnp.array(y_data)


def predict(params, x):
    """y = tanh(w1 * x) + tanh(w2 * x)"""
    w1, w2 = params[0], params[1]
    return jnp.tanh(w1 * x) + jnp.tanh(w2 * x)


def loss_fn(params):
    pred = predict(params, x_jax)
    return jnp.mean((pred - y_jax) ** 2)


grad_fn = jax.grad(loss_fn)

# ---------------------------------------------------------------------------
# Compute loss landscape on a grid
# ---------------------------------------------------------------------------
w1_range = np.linspace(-3, 3, 300)
w2_range = np.linspace(-3, 3, 300)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        Z[i, j] = float(loss_fn(jnp.array([W1[i, j], W2[i, j]])))

# ---------------------------------------------------------------------------
# Run basin filling
# ---------------------------------------------------------------------------
initial = jnp.array([0.3, 2.0])

create_optimizer = lambda: optax.sgd(learning_rate=0.01, momentum=0.9)

cfg = BasinFillingConfig(
    detect_stuck=detection.loss_plateau(threshold=1e-6),
    estimate_radius=radius.fixed(value=0.8),
    check_inside=containment.sphere_check(),
    walkthrough=walkthrough.saddle_direction(loss_fn, step_size=0.1),
    detection_patience=50, max_runs=12, max_steps_per_run=5000,
)

result = run_basin_filling(loss_fn, grad_fn, initial, create_optimizer, cfg)

# Build run_data from result (need archive snapshots per run)
archive = []
run_data = []
traj_idx = 0
for traj in result['trajectories']:
    archive_before = list(archive)
    # Check if this run found a basin (trajectory ends with stuck detection)
    found = None
    for b in result['archive']:
        if b.run_idx == traj_idx:
            found = b
            break
    if found is not None:
        archive.append(found)
        run_data.append((traj, found, archive_before))
    traj_idx += 1

# Pick first few distinct basins for panels
seen = set()
panels = []
for traj, basin, arch_before in run_data:
    key = (round(basin.center[0], 1), round(basin.center[1], 1))
    if key not in seen:
        seen.add(key)
        panels.append((traj, basin, arch_before))
    if len(panels) == 4:
        break

n_panels = min(len(panels), 4)
if n_panels == 0:
    print("No basins found!")
    sys.exit(1)

# Pad to 4 if needed
while len(panels) < 4:
    panels.append(panels[-1])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 9.5), constrained_layout=True)
axes = axes.flatten()

for idx, ax in enumerate(axes):
    traj, basin, arch_before = panels[idx]

    # Loss contours
    ax.contourf(W1, W2, Z, levels=np.linspace(0, 1.0, 30),
                cmap='bone_r', alpha=0.6)
    ax.contour(W1, W2, Z, levels=np.linspace(0, 1.0, 15),
               colors='gray', linewidths=0.3)

    # Filled basins — Phi damping heatmap
    if arch_before:
        Phi = np.ones_like(Z)
        for b in arch_before:
            dist_sq = (W1 - b.center[0])**2 + (W2 - b.center[1])**2
            psi_k = np.exp(-dist_sq / (2.0 * b.radius**2))
            Phi *= (1.0 - psi_k)
        ax.contourf(W1, W2, 1.0 - Phi,
                    levels=[0.05, 0.2, 0.5, 0.8, 0.95],
                    cmap='Blues', alpha=0.35, zorder=2)
        ax.contour(W1, W2, 1.0 - Phi,
                   levels=[0.5], colors='steelblue',
                   linewidths=1.0, linestyles='--', zorder=2)
        for b in arch_before:
            ax.plot(b.center[0], b.center[1], 'x', color='steelblue',
                    markersize=7, markeredgewidth=1.5, zorder=5)

    # Trajectory
    pts = np.array([s.params for s in traj])
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#d62728',
            linewidth=0.8, alpha=0.7, zorder=3)

    # Start and end
    ax.plot(pts[0, 0], pts[0, 1], 'o', color='white', markersize=6,
            markeredgecolor='black', markeredgewidth=1.2, zorder=6)
    ax.plot(pts[-1, 0], pts[-1, 1], '*', color='#d62728', markersize=12,
            markeredgecolor='black', markeredgewidth=0.5, zorder=6)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    n_filled = len(arch_before)
    loss_val = basin.loss
    ax.set_title(f'Run {idx+1}:  ({basin.center[0]:.2f}, {basin.center[1]:.2f})'
                 f'  loss={loss_val:.4f}  [{n_filled} filled]',
                 fontsize=9)
    ax.set_xlabel(r'$w_1$', fontsize=9)
    ax.set_ylabel(r'$w_2$', fontsize=9)
    ax.tick_params(labelsize=8)

fig.suptitle(
    r'Basin filling on $y = \tanh(w_1 x) + \tanh(w_2 x)$'
    '\n'
    r'$r\!=\!0.8$, saddle direction walkthrough'
    r'$\quad\bigstar$ found'
    r'$\quad\bigcirc$ start',
    fontsize=10,
)

outpath = os.path.join(os.path.dirname(__file__), 'basin_filling_toynet.pdf')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
outpath_png = outpath.replace('.pdf', '.png')
fig.savefig(outpath_png, dpi=200, bbox_inches='tight')
print(f'Saved: {outpath}')
print(f'Saved: {outpath_png}')

# Print summary
print(f'\nFound {len(archive)} basins:')
for i, b in enumerate(archive):
    pred = predict(jnp.array(b.center), x_jax)
    mse = float(jnp.mean((pred - y_jax) ** 2))
    print(f'  Basin {i}: w=({b.center[0]:.3f}, {b.center[1]:.3f}), '
          f'loss={b.loss:.6f}, MSE={mse:.6f}')
plt.close()
