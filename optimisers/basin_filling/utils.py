"""Shared utilities for basin-filling strategies."""

import numpy as np
import jax.numpy as jnp


def run_dependent_direction(params, run_idx):
    """Deterministic unit direction that varies with *run_idx*.

    For 2-D: evenly spaced angles around the unit circle.
    For n-D: rotation in the (0, 1) coordinate plane.
    """
    n = params.shape[0] if hasattr(params, 'shape') else len(params)
    angle = 2 * np.pi * run_idx / max(run_idx + 1, 2)
    d = np.zeros(n)
    d[0] = np.cos(angle)
    d[1 % n] += np.sin(angle)
    return jnp.array(d / np.linalg.norm(d))
