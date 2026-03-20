"""
Core basin-filling algorithm with pluggable strategies.

The main loop is strategy-agnostic: detection, containment, walk-through,
and radius estimation are all injected via callables on BasinFillingConfig.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Basin:
    """A discovered basin in the loss landscape."""
    center: np.ndarray
    loss: float
    radius: float
    run_idx: int = -1
    extra: dict = field(default_factory=dict)


@dataclass
class StepRecord:
    """Record of a single optimization step (for trajectory logging)."""
    params: np.ndarray
    loss: float
    grad_norm: float
    walkthrough: bool
    basin_idx: int = -1


@dataclass
class StepInfo:
    """Snapshot of all information available at an optimization step.

    Strategies receive this as their first argument, so the interface
    remains stable as new strategies are added.
    """
    params: jnp.ndarray
    loss: float
    grads: jnp.ndarray
    grad_norm: float
    velocity: jnp.ndarray
    run_idx: int
    step_idx: int
    loss_history: List[float]
    trajectory: List[StepRecord]


@dataclass
class BasinFillingConfig:
    """Configuration with pluggable strategy callables.

    Strategy signatures::

        detect_stuck(info: StepInfo, stuck_count: int) -> int
        check_inside(params: ndarray, archive: List[Basin]) -> Optional[int]
        walkthrough(info: StepInfo, basin: Basin) -> ndarray  # full step vector
        estimate_radius(info: StepInfo) -> float

    Use factory functions from ``basin_filling.strategies`` to create them.
    Pass ``None`` for any strategy to get the default.
    """
    detect_stuck: Optional[Callable] = None
    check_inside: Optional[Callable] = None
    walkthrough: Optional[Callable] = None
    estimate_radius: Optional[Callable] = None

    detection_patience: int = 50
    max_runs: int = 30
    max_steps_per_run: int = 1000
    max_walkthrough_steps: int = 500

    def __post_init__(self):
        from .strategies import detection, containment, walkthrough, radius
        if self.detect_stuck is None:
            self.detect_stuck = detection.loss_plateau()
        if self.check_inside is None:
            self.check_inside = containment.sphere_check()
        if self.walkthrough is None:
            self.walkthrough = walkthrough.momentum()
        if self.estimate_radius is None:
            self.estimate_radius = radius.loss_contour()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_basin_filling(
    loss_fn: Callable,
    grad_fn: Callable,
    initial_params: jnp.ndarray,
    create_optimizer: Callable[[], optax.GradientTransformation],
    config: Optional[BasinFillingConfig] = None,
) -> dict:
    """Run the basin-filling meta-optimization.

    Returns dict with 'archive', 'trajectories', 'best_basin', 'num_runs'.
    """
    cfg = config or BasinFillingConfig()
    archive: List[Basin] = []
    trajectories: List[List[StepRecord]] = []

    for run_idx in range(cfg.max_runs):
        traj, basin = _single_run(
            loss_fn, grad_fn, initial_params,
            create_optimizer, archive, run_idx, cfg,
        )
        trajectories.append(traj)

        if basin is not None:
            archive.append(basin)
            # Adaptive radius hook (backward-compatible)
            hook = getattr(cfg.estimate_radius, 'post_run_hook', None)
            if hook is not None:
                hook(archive, traj)
        else:
            break

    best = min(archive, key=lambda b: b.loss) if archive else None
    return {
        'archive': archive,
        'trajectories': trajectories,
        'best_basin': best,
        'num_runs': len(trajectories),
    }


def run_random_restarts(
    loss_fn: Callable,
    grad_fn: Callable,
    create_optimizer: Callable[[], optax.GradientTransformation],
    num_restarts: int,
    max_steps: int,
    bounds: tuple = (-5.0, 5.0),
    ndim: int = 2,
    seed: int = 42,
) -> dict:
    """Random restarts baseline for comparison."""
    rng = np.random.RandomState(seed)
    results = []

    for _ in range(num_restarts):
        init = jnp.array(rng.uniform(bounds[0], bounds[1], size=ndim))
        optimizer = create_optimizer()
        opt_state = optimizer.init(init)
        params = init

        for _ in range(max_steps):
            grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            if jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                break

        final_loss = float(loss_fn(params))
        results.append({'params': np.array(params), 'loss': final_loss})

    best = min(results, key=lambda r: r['loss'])
    return {
        'results': results,
        'best_params': best['params'],
        'best_loss': best['loss'],
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _single_run(loss_fn, grad_fn, initial_params, create_optimizer,
                archive, run_idx, cfg):
    """One optimization pass: normal steps + walk-throughs."""
    optimizer = create_optimizer()
    opt_state = optimizer.init(initial_params)
    params = jnp.array(initial_params)

    trajectory: List[StepRecord] = []
    loss_history: List[float] = []
    stuck_count = 0

    velocity = jnp.zeros_like(params)
    vel_decay = 0.9

    need_reinit = False
    consecutive_wt = 0

    for step_idx in range(cfg.max_steps_per_run):
        loss_val = float(loss_fn(params))
        grads = grad_fn(params)
        grad_norm = float(jnp.linalg.norm(grads))

        if jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
            break

        info = StepInfo(
            params=params, loss=loss_val, grads=grads,
            grad_norm=grad_norm, velocity=velocity,
            run_idx=run_idx, step_idx=step_idx,
            loss_history=loss_history, trajectory=trajectory,
        )

        basin_idx = cfg.check_inside(params, archive)

        if basin_idx is not None:
            # --- Walk through known basin ---
            consecutive_wt += 1
            if consecutive_wt > cfg.max_walkthrough_steps:
                break

            update = cfg.walkthrough(info, archive[basin_idx])
            params = params + update
            direction = update / (jnp.linalg.norm(update) + 1e-12)
            velocity = vel_decay * velocity + (1 - vel_decay) * direction
            need_reinit = True
            stuck_count = 0

            trajectory.append(StepRecord(
                params=np.array(params), loss=loss_val,
                grad_norm=grad_norm, walkthrough=True, basin_idx=basin_idx,
            ))
        else:
            # --- Normal optimization ---
            consecutive_wt = 0

            if need_reinit:
                optimizer = create_optimizer()
                opt_state = optimizer.init(params)
                need_reinit = False

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            velocity = vel_decay * velocity + (1 - vel_decay) * (-grads)

            loss_history.append(loss_val)
            stuck_count = cfg.detect_stuck(info, stuck_count)

            trajectory.append(StepRecord(
                params=np.array(params), loss=loss_val,
                grad_norm=grad_norm, walkthrough=False,
            ))

            if stuck_count >= cfg.detection_patience:
                radius = cfg.estimate_radius(info)
                extra = getattr(info, '_hessian_extra', {})
                basin = Basin(
                    center=np.array(params), loss=loss_val,
                    radius=radius, run_idx=run_idx, extra=extra,
                )
                return trajectory, basin

    return trajectory, None
