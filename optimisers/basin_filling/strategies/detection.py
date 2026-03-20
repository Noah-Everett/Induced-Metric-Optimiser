"""
Detection strategies: determine when the optimizer is stuck in a basin.

Each factory returns a callable with signature::

    detect(info: StepInfo, stuck_count: int) -> int

Heuristic: loss_plateau, gradient_norm
Geometric: function_space_speed, geodesic_stagnation
"""

import jax
import jax.numpy as jnp


# ===================================================================
# Heuristic
# ===================================================================

def loss_plateau(threshold=1e-7):
    """Stuck when consecutive loss improvements are below *threshold*."""
    def detect(info, stuck_count):
        h = info.loss_history
        if len(h) < 2:
            return 0
        improvement = abs(h[-2] - h[-1])
        return stuck_count + 1 if improvement < threshold else 0
    return detect


def gradient_norm(threshold=1e-6):
    """Stuck when the gradient norm is below *threshold*."""
    def detect(info, stuck_count):
        return stuck_count + 1 if info.grad_norm < threshold else 0
    return detect


# ===================================================================
# Geometric
# ===================================================================

def function_space_speed(probe_fn, threshold=1e-6):
    r"""Stuck when model behaviour stops changing: v_func = ||J dtheta|| -> 0.

    Uses a single JVP through *probe_fn*.  For scalar loss functions
    (2-D benchmarks), pass the loss function itself as *probe_fn*.
    """
    def detect(info, stuck_count):
        traj = info.trajectory
        if len(traj) < 1:
            return 0
        prev_params = jnp.array(traj[-1].params)
        delta = info.params - prev_params
        if float(jnp.linalg.norm(delta)) < 1e-30:
            return stuck_count + 1
        _, J_delta = jax.jvp(probe_fn, (info.params,), (delta,))
        v_func = float(jnp.linalg.norm(jnp.atleast_1d(J_delta)))
        return stuck_count + 1 if v_func < threshold else 0
    return detect


def geodesic_stagnation(threshold=1e-4, window=10):
    r"""Stuck when Riemannian displacement fails to produce loss improvement.

    Uses the induced metric G = I + gg^T from the paper.  Cost: O(N)
    per step (two dot products, no JVPs).
    """
    state = {'arc_lengths': []}

    def detect(info, stuck_count):
        traj = info.trajectory
        if len(traj) < 1:
            state['arc_lengths'] = []
            return 0
        prev_params = jnp.array(traj[-1].params)
        delta = info.params - prev_params
        delta_norm_sq = float(jnp.sum(delta ** 2))
        g_dot_delta = float(jnp.sum(info.grads * delta))
        riem_speed = (delta_norm_sq + g_dot_delta ** 2) ** 0.5
        state['arc_lengths'].append(riem_speed)

        h = info.loss_history
        if len(state['arc_lengths']) < window or len(h) < window:
            return 0
        arc_total = sum(state['arc_lengths'][-window:])
        loss_change = abs(h[-window] - h[-1])
        if arc_total < 1e-30:
            return stuck_count + 1
        eta = loss_change / arc_total
        return stuck_count + 1 if eta < threshold else 0
    return detect
