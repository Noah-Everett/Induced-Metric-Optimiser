# optimisers/jax_diag_utils.py
"""
Shared JAX utilities for diagonal metric operations.

Used by jax_learnable_diag.py, jax_learnable_diag_curv.py, and
jax_derived_newton_diag.py.
"""

import jax
import jax.numpy as jnp


def apply_diag(log_diag, x):
    """Apply diagonal inverse metric diag(exp(log_diag)) to x, per leaf."""
    return jax.tree.map(lambda s, t: jnp.exp(s) * t, log_diag, x)


def apply_diag_softplus(log_diag, x):
    """Apply diagonal inverse metric diag(softplus(log_diag)) to x, per leaf."""
    return jax.tree.map(lambda s, t: jax.nn.softplus(s) * t, log_diag, x)


def mean_center(log_diag):
    """Per-leaf mean-centre s to control global scale."""
    return jax.tree.map(lambda s: s - jnp.mean(s), log_diag)


def clip(log_diag, lo, hi):
    """Per-leaf clip s values to [lo, hi]."""
    return jax.tree.map(lambda s: jnp.clip(s, lo, hi), log_diag)
