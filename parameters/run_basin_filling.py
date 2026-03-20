"""
Basin-filling optimizer experiments on 2D test functions.

Usage::

    python3 run_basin_filling.py --function himmelblau
    python3 run_basin_filling.py --all
    python3 run_basin_filling.py --function rastrigin --walkthrough entry
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'optimisers'))

from sweep_small_examples import TEST_FUNCTIONS
from basin_filling import BasinFillingConfig, run_basin_filling, run_random_restarts
from basin_filling.strategies import detection, containment, walkthrough, radius

# ---------------------------------------------------------------------------
# Plot ranges per function
# ---------------------------------------------------------------------------

PLOT_RANGES = {
    'beale':      ((-4.5, 4.5),   (-4.5, 4.5)),
    'rosenbrock': ((-2.5, 3.0),   (-1.5, 3.5)),
    'himmelblau': ((-5.0, 5.0),   (-5.0, 5.0)),
    'ackley':     ((-5.0, 5.0),   (-5.0, 5.0)),
    'rastrigin':  ((-5.12, 5.12), (-5.12, 5.12)),
}

# ---------------------------------------------------------------------------
# Strategy CLI mapping
# ---------------------------------------------------------------------------

DETECTION_MAP = {
    'loss_plateau': lambda a, **kw: detection.loss_plateau(threshold=a.detection_threshold),
    'gradient_norm': lambda a, **kw: detection.gradient_norm(threshold=a.detection_threshold),
    'function_space_speed': lambda a, loss_fn=None, **kw: detection.function_space_speed(
        probe_fn=loss_fn, threshold=a.detection_threshold),
    'geodesic_stagnation': lambda a, **kw: detection.geodesic_stagnation(
        threshold=a.detection_threshold, window=10),
}

WALKTHROUGH_MAP = {
    'momentum': lambda a, **kw: walkthrough.momentum(step_size=a.walkthrough_step_size),
    'entry': lambda a, **kw: walkthrough.entry(step_size=a.walkthrough_step_size),
    'anti_gradient': lambda a, **kw: walkthrough.anti_gradient(step_size=a.walkthrough_step_size),
    'random': lambda a, **kw: walkthrough.random_direction(step_size=a.walkthrough_step_size),
    'saddle': lambda a, loss_fn=None, **kw: walkthrough.saddle_direction(
        loss_fn, step_size=a.walkthrough_step_size),
    'gradient_orthogonal': lambda a, loss_fn=None, **kw: walkthrough.gradient_orthogonal(
        loss_fn, step_size=a.walkthrough_step_size),
    'conjugate': lambda a, loss_fn=None, **kw: walkthrough.conjugate_basin_escape(
        loss_fn, step_size=a.walkthrough_step_size),
}

EXTENT_MAP = {
    'fixed': lambda a, **kw: radius.fixed(value=a.fixed_radius),
    'trajectory': lambda a, **kw: radius.trajectory(scale=a.trajectory_scale),
    'loss_contour': lambda a, **kw: radius.loss_contour(
        factor=a.loss_contour_factor, scale=a.trajectory_scale),
    'hessian': lambda a, loss_fn=None, **kw: radius.hessian_ellipsoid(loss_fn),
    'adaptive': lambda a, **kw: radius.adaptive(initial_radius=a.fixed_radius),
}

CONTAINMENT_MAP = {
    'sphere': lambda a, **kw: containment.sphere_check(),
    'ellipsoid': lambda a, **kw: containment.ellipsoid_check(),
    'loss_gated': lambda a, loss_fn=None, **kw: containment.loss_gated_check(loss_fn),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_optimizer(lr=0.01, mom=0.9):
    return lambda: optax.sgd(learning_rate=lr, momentum=mom)


def count_distinct(points, threshold=0.1):
    if len(points) == 0:
        return 0
    clusters = [points[0]]
    for p in points[1:]:
        if all(np.linalg.norm(p - c) > threshold for c in clusters):
            clusters.append(p)
    return len(clusters)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(func_name, func, bf_result, rr_result, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    xr, yr = PLOT_RANGES[func_name]

    xs = np.linspace(*xr, 300)
    ys = np.linspace(*yr, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = float(func(jnp.array([X[i, j], Y[i, j]])))
    Z_log = np.log10(np.clip(Z, 1e-10, None))
    lo, hi = Z_log.min(), min(Z_log.max(), Z_log.min() + 6)
    levels = np.linspace(lo, hi, 30)

    # -- left: basin filling --
    ax = axes[0]
    ax.contour(X, Y, Z_log, levels=levels, cmap='terrain', alpha=0.6)
    ax.contourf(X, Y, Z_log, levels=levels, cmap='terrain', alpha=0.2)

    cmap = plt.cm.tab10
    n_runs = max(len(bf_result['trajectories']), 1)
    for run_idx, traj in enumerate(bf_result['trajectories']):
        if not traj:
            continue
        pts = np.array([s.params for s in traj])
        wt = np.array([s.walkthrough for s in traj])
        colour = cmap(run_idx / n_runs)
        mask_n = ~wt
        if np.any(mask_n):
            ax.plot(pts[mask_n, 0], pts[mask_n, 1], '-',
                    color=colour, linewidth=1.0, alpha=0.8)
        if np.any(wt):
            ax.plot(pts[wt, 0], pts[wt, 1], '--',
                    color=colour, linewidth=1.5, alpha=0.6)

    for basin in bf_result['archive']:
        is_best = (bf_result['best_basin'] is not None
                   and np.allclose(basin.center, bf_result['best_basin'].center))
        col = 'gold' if is_best else 'red'
        lw = 2.5 if is_best else 1.5
        ax.add_patch(plt.Circle(basin.center, basin.radius,
                                fill=False, color=col, linewidth=lw))
        ax.plot(*basin.center, 'x', color=col, markersize=8, markeredgewidth=2)
        ax.annotate(f'{basin.loss:.2e}', basin.center,
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=7, color=col)

    init = TEST_FUNCTIONS[func_name]['initial']
    ax.plot(float(init[0]), float(init[1]), 's', color='white',
            markersize=10, markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax.set_xlim(xr); ax.set_ylim(yr)
    ax.set_title(f'Basin Filling  ({bf_result["num_runs"]} runs, '
                 f'{len(bf_result["archive"])} basins)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(handles=[
        Line2D([0], [0], color='gray', lw=1, label='Normal'),
        Line2D([0], [0], color='gray', lw=1.5, ls='--', label='Walkthrough'),
        Line2D([0], [0], marker='x', color='red', ls='None', ms=8, label='Basin'),
        Line2D([0], [0], marker='x', color='gold', ls='None', ms=8, label='Best'),
        Line2D([0], [0], marker='s', color='white', markeredgecolor='black',
               ls='None', ms=8, label='Start'),
    ], loc='upper right', fontsize=8)

    # -- right: random restarts --
    ax = axes[1]
    ax.contour(X, Y, Z_log, levels=levels, cmap='terrain', alpha=0.6)
    ax.contourf(X, Y, Z_log, levels=levels, cmap='terrain', alpha=0.2)
    for r in rr_result['results']:
        ax.plot(*r['params'], 'o', color='blue', markersize=5, alpha=0.5)
    ax.plot(*rr_result['best_params'], '*', color='gold', markersize=15,
            markeredgecolor='black', markeredgewidth=1, zorder=5)
    ax.set_xlim(xr); ax.set_ylim(yr)
    ax.set_title(f'Random Restarts  ({len(rr_result["results"])} runs)')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.suptitle(f'{func_name.capitalize()} -- Basin Filling vs Random Restarts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(func_name, config, lr=0.01, mom=0.9):
    func_info = TEST_FUNCTIONS[func_name]
    loss_fn, grad_fn = func_info['func'], func_info['grad']
    initial = func_info['initial']
    create_opt = make_optimizer(lr, mom)

    print(f'\n{"=" * 60}')
    print(f'  {func_name.upper()}')
    print(f'{"=" * 60}')

    print(f'  Running basin filling (max {config.max_runs} runs) ...')
    t0 = time.time()
    bf = run_basin_filling(loss_fn, grad_fn, initial, create_opt, config)
    bf_time = time.time() - t0
    total_steps = sum(len(t) for t in bf['trajectories'])

    print(f'  Basin filling: {bf["num_runs"]} runs, '
          f'{len(bf["archive"])} basins, '
          f'{total_steps} total steps, {bf_time:.2f}s')
    if bf['best_basin']:
        b = bf['best_basin']
        print(f'  Best basin: loss={b.loss:.6e}  '
              f'at ({b.center[0]:.4f}, {b.center[1]:.4f})')
    for i, basin in enumerate(bf['archive']):
        print(f'    Basin {i}: loss={basin.loss:.6e}, r={basin.radius:.4f}, '
              f'center=({basin.center[0]:.4f}, {basin.center[1]:.4f})')

    steps_per = config.max_steps_per_run
    num_rr = max(total_steps // steps_per, bf['num_runs'])
    print(f'  Running {num_rr} random restarts ...')
    t0 = time.time()
    rr = run_random_restarts(
        loss_fn, grad_fn, create_opt,
        num_restarts=num_rr, max_steps=steps_per,
        bounds=PLOT_RANGES[func_name][0], ndim=2,
    )
    rr_time = time.time() - t0
    distinct = count_distinct(
        np.array([r['params'] for r in rr['results']]), threshold=0.1)
    print(f'  Random restarts: best={rr["best_loss"]:.6e}  '
          f'at ({rr["best_params"][0]:.4f}, {rr["best_params"][1]:.4f}), '
          f'{distinct} distinct, {rr_time:.2f}s')

    results_dir = os.path.join(_HERE, '..', 'results', 'basin_filling')
    os.makedirs(results_dir, exist_ok=True)
    plot_results(func_name, loss_fn, bf, rr,
                 os.path.join(results_dir, f'{func_name}.png'))
    return bf, rr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Basin-filling optimizer on 2D test functions')
    p.add_argument('--function', type=str, default=None,
                   choices=list(TEST_FUNCTIONS.keys()))
    p.add_argument('--all', action='store_true')

    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)

    p.add_argument('--detection', default='loss_plateau',
                   choices=list(DETECTION_MAP))
    p.add_argument('--detection_threshold', type=float, default=1e-7)
    p.add_argument('--detection_patience', type=int, default=50)
    p.add_argument('--walkthrough', default='momentum',
                   choices=list(WALKTHROUGH_MAP))
    p.add_argument('--walkthrough_step_size', type=float, default=0.05)
    p.add_argument('--extent', default='loss_contour',
                   choices=list(EXTENT_MAP))
    p.add_argument('--containment', default='sphere',
                   choices=list(CONTAINMENT_MAP))
    p.add_argument('--fixed_radius', type=float, default=0.5)
    p.add_argument('--trajectory_scale', type=float, default=1.0)
    p.add_argument('--loss_contour_factor', type=float, default=0.1)
    p.add_argument('--max_runs', type=int, default=30)
    p.add_argument('--max_steps', type=int, default=1000)

    args = p.parse_args()

    functions = (list(TEST_FUNCTIONS.keys())
                 if (args.all or args.function is None)
                 else [args.function])

    for fn in functions:
        # Geometric strategies need loss_fn, resolved per function
        loss_fn = TEST_FUNCTIONS[fn]['func']
        kw = dict(loss_fn=loss_fn)

        config = BasinFillingConfig(
            detect_stuck=DETECTION_MAP[args.detection](args, **kw),
            check_inside=CONTAINMENT_MAP[args.containment](args, **kw),
            walkthrough=WALKTHROUGH_MAP[args.walkthrough](args, **kw),
            estimate_radius=EXTENT_MAP[args.extent](args, **kw),
            detection_patience=args.detection_patience,
            max_runs=args.max_runs,
            max_steps_per_run=args.max_steps,
        )
        run_experiment(fn, config, args.lr, args.momentum)


if __name__ == '__main__':
    main()
