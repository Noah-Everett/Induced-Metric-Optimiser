"""
Basin filling via spatially varying off-diagonal induced metric.

The induced metric g = gamma + xi * (a l^T + l b^T + l l^T) is modified
between runs by promoting xi, a, b to spatially varying fields:

    xi(theta)  = xi_0 * Phi(theta)          -- removes damping near basins
    a(theta)   = a_0 + sum_k alpha_k psi_k n_k  -- steers away from basins
    b(theta)   = a(theta)                    -- symmetric case

The update has two components::

    delta_theta = -eta * [ (1 + xi c_al) / D * l/gamma  -  xi c_ll / D * a/gamma ]

    gradient component (speed)  +  steering component (direction)

Key properties:
- Rank-2 Woodbury: O(KN) per step regardless of variant.
- Positive-definite in the symmetric case (a = b).
- No new critical points.  No loss modification.
- Reduces to the paper's fixed diagonal variant when archive is empty.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from .core import Basin, StepRecord, StepInfo


@dataclass
class MetricBasinFillingConfig:
    """Configuration for metric-modification basin filling.

    Strategy callables (same interface as core.py)::

        detect_stuck(info: StepInfo, stuck_count: int) -> int
        estimate_radius(info: StepInfo) -> float

    Containment and walkthrough are handled by the metric.
    """
    detect_stuck: Optional[Callable] = None
    estimate_radius: Optional[Callable] = None

    lr: float = 0.01
    xi: float = 0.1
    beta_m: float = 0.9
    gamma: float = 1.0

    # Off-diagonal steering
    steering_strength: float = 1.0   # alpha_0: global steering intensity
    steering_eps: float = 1e-8       # regularisation for n_k at basin center

    detection_patience: int = 50
    max_runs: int = 30
    max_steps_per_run: int = 1000

    def __post_init__(self):
        from .strategies import detection, radius
        if self.detect_stuck is None:
            self.detect_stuck = detection.loss_plateau()
        if self.estimate_radius is None:
            self.estimate_radius = radius.loss_contour()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metric_fields(params, grads, archive, cfg):
    """Compute Phi(theta), xi(theta), a(theta) for the current position.

    Returns (phi, xi, a_vec) where a_vec is the N-dimensional steering vector.
    """
    N = params.shape[0]
    phi = 1.0
    a_vec = jnp.zeros(N)

    for basin in archive:
        diff = params - jnp.array(basin.center)
        dist_sq = float(jnp.sum(diff ** 2))
        dist = np.sqrt(dist_sq)

        psi_k = np.exp(-dist_sq / (2.0 * basin.radius ** 2))
        phi *= (1.0 - psi_k)

        # Steering direction: unit radial outward from basin center
        n_k = diff / (dist + cfg.steering_eps)

        # Steering strength scales with basin radius
        alpha_k = cfg.steering_strength * basin.radius
        a_vec = a_vec + alpha_k * psi_k * n_k

    xi = cfg.xi * phi
    return phi, xi, a_vec


def _offdiag_update(d, grads, a_vec, xi, gamma):
    """Compute g'^{-1} d via Woodbury on g' = gamma I + xi (a l^T + l a^T + l l^T).

    Symmetric case (a = b).  Returns the preconditioned direction.
    """
    inv_gamma = 1.0 / gamma

    # If xi is negligible, skip the Woodbury
    if abs(xi) < 1e-15:
        return inv_gamma * d

    l = grads  # shorthand

    # Scalar contractions
    c_al = float(jnp.dot(a_vec, l)) * inv_gamma
    c_ll = float(jnp.dot(l, l)) * inv_gamma
    c_aa = float(jnp.dot(a_vec, a_vec)) * inv_gamma

    # Determinant D (symmetric case: c_bl = c_al, c_ba = c_aa)
    D = (1.0
         + xi * (2.0 * c_al + c_ll)
         + xi ** 2 * (c_al ** 2 - c_ll * c_aa))

    if abs(D) < 1e-15:
        return inv_gamma * d

    # Full Woodbury solve for general direction d
    # g'^{-1} d via U = [a, l], V = [l, a+l] (symmetric: b = a)
    # M = I_2 + (xi/gamma) V^T U
    alpha_w = xi * inv_gamma

    # V^T d  (2-vector)
    vt_d_0 = float(jnp.dot(l, d)) * inv_gamma        # l^T d / gamma
    vt_d_1 = float(jnp.dot(a_vec + l, d)) * inv_gamma # (a+l)^T d / gamma

    # M matrix
    M00 = 1.0 + alpha_w * c_al
    M01 = alpha_w * c_ll
    M10 = alpha_w * (c_aa + c_al)
    M11 = 1.0 + alpha_w * (c_al + c_ll)

    # Solve M w = V^T (d/gamma) via Cramer's rule
    det_M = M00 * M11 - M01 * M10
    w0 = (M11 * vt_d_0 - M01 * vt_d_1) / det_M
    w1 = (M00 * vt_d_1 - M10 * vt_d_0) / det_M

    # g'^{-1} d = d/gamma - (xi/gamma) * U w
    result = inv_gamma * d - alpha_w * (w0 * a_vec + w1 * l)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_metric_basin_filling(
    loss_fn: Callable,
    grad_fn: Callable,
    initial_params: jnp.ndarray,
    config: Optional[MetricBasinFillingConfig] = None,
) -> dict:
    """Run basin filling with spatially varying off-diagonal metric.

    Unlike core.run_basin_filling, this does NOT take a create_optimizer
    argument -- the optimizer IS the metric-modified update rule.

    Returns dict with 'archive', 'trajectories', 'best_basin', 'num_runs'.
    """
    cfg = config or MetricBasinFillingConfig()
    archive: List[Basin] = []
    trajectories: List[List[StepRecord]] = []

    for run_idx in range(cfg.max_runs):
        traj, basin = _single_run(
            loss_fn, grad_fn, initial_params, archive, run_idx, cfg,
        )
        trajectories.append(traj)

        if basin is not None:
            archive.append(basin)
        else:
            break

    best = min(archive, key=lambda b: b.loss) if archive else None
    return {
        'archive': archive,
        'trajectories': trajectories,
        'best_basin': best,
        'num_runs': len(trajectories),
    }


def _single_run(loss_fn, grad_fn, initial_params, archive, run_idx, cfg):
    """One optimization pass with the off-diagonal metric-modified update."""
    params = jnp.array(initial_params)

    trajectory: List[StepRecord] = []
    loss_history: List[float] = []
    stuck_count = 0

    momentum = jnp.zeros_like(params)

    for step_idx in range(cfg.max_steps_per_run):
        loss_val = float(loss_fn(params))
        grads = grad_fn(params)
        grad_norm = float(jnp.linalg.norm(grads))

        if jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
            break

        # Compute spatially varying metric fields
        phi, xi, a_vec = _compute_metric_fields(params, grads, archive, cfg)

        # Riemannian gradient: g'^{-1} grads (precondition FIRST)
        riem_grad = _offdiag_update(grads, grads, a_vec, xi, cfg.gamma)

        # Momentum on the preconditioned gradient (matches optax pattern)
        momentum = cfg.beta_m * momentum + (1.0 - cfg.beta_m) * riem_grad
        bc = 1.0 - cfg.beta_m ** (step_idx + 1)
        m_hat = momentum / bc

        update = -cfg.lr * m_hat
        params = params + update

        # Record (phi < 0.5 flags "inside a filled basin" for visualisation)
        trajectory.append(StepRecord(
            params=np.array(params), loss=loss_val,
            grad_norm=grad_norm, walkthrough=(phi < 0.5),
        ))
        loss_history.append(loss_val)

        info = StepInfo(
            params=params, loss=loss_val, grads=grads,
            grad_norm=grad_norm,
            velocity=m_hat,
            run_idx=run_idx, step_idx=step_idx,
            loss_history=loss_history, trajectory=trajectory,
        )

        stuck_count = cfg.detect_stuck(info, stuck_count)

        if stuck_count >= cfg.detection_patience:
            r = cfg.estimate_radius(info)
            basin = Basin(
                center=np.array(params), loss=loss_val,
                radius=r, run_idx=run_idx,
            )
            return trajectory, basin

    return trajectory, None
