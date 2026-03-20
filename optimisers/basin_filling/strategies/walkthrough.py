"""
Walk-through strategies: how the optimizer moves through a known basin.

Each factory returns a callable with signature::

    walk(info: StepInfo, basin: Basin) -> ndarray  (full step vector)

Heuristic: momentum, entry, anti_gradient, random_direction
Geometric: saddle_direction, gradient_orthogonal, conjugate_basin_escape
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import run_dependent_direction


# ===================================================================
# Heuristic
# ===================================================================

def momentum(step_size=0.05):
    """Continue in the velocity direction; fall back to run-dependent angle."""
    def walk(info, basin):
        norm = float(jnp.linalg.norm(info.velocity))
        if norm > 1e-12:
            direction = info.velocity / norm
        else:
            direction = run_dependent_direction(info.params, info.run_idx)
        return step_size * direction
    return walk


def entry(step_size=0.05):
    """Walk radially outward from the basin center."""
    def walk(info, basin):
        d = np.array(info.params) - basin.center
        dnorm = np.linalg.norm(d)
        if dnorm > 1e-12:
            direction = jnp.array(d / dnorm)
        else:
            direction = run_dependent_direction(info.params, info.run_idx)
        return step_size * direction
    return walk


def anti_gradient(step_size=0.05):
    """Walk uphill on the original loss to escape the basin."""
    def walk(info, basin):
        norm = float(jnp.linalg.norm(info.grads))
        if norm > 1e-12:
            direction = info.grads / norm
        else:
            direction = run_dependent_direction(info.params, info.run_idx)
        return step_size * direction
    return walk


def random_direction(step_size=0.05):
    """Fixed random direction per basin (deterministic from basin center)."""
    def walk(info, basin):
        seed = int(np.sum(np.abs(basin.center) * 1e4)) % (2**31)
        key = jax.random.PRNGKey(seed)
        d = jax.random.normal(key, shape=info.params.shape)
        return step_size * (d / jnp.linalg.norm(d))
    return walk


# ===================================================================
# Geometric
# ===================================================================

def saddle_direction(loss_fn, step_size=0.05, num_iters=20,
                     scale_by_radius=True):
    r"""Walk toward the nearest saddle via the softest Hessian eigenvector.

    At the basin center, the smallest Hessian eigenvalue direction points
    toward the nearest index-1 saddle.  Crossing the saddle enters an
    adjacent basin's gradient flow -- a topological guarantee.

    Cost: *num_iters* HVPs per basin, cached.
    """
    _cache = {}

    def _hvp(params, v):
        return jax.jvp(jax.grad(loss_fn), (params,), (v,))[1]

    def _min_eigvec(params, n, key):
        v = jax.random.normal(key, shape=(n,))
        v = v / jnp.linalg.norm(v)
        for _ in range(num_iters):
            hv = _hvp(params, v)
            neg_hv = -hv
            norm = jnp.linalg.norm(neg_hv)
            v = jnp.where(norm > 1e-30, neg_hv / norm, v)
        return v

    def walk(info, basin):
        cache_key = basin.center.tobytes()
        if cache_key not in _cache:
            center = jnp.array(basin.center)
            n = center.shape[0]
            seed = int(np.sum(np.abs(basin.center) * 1e4)) % (2**31)
            eigvec = _min_eigvec(center, n, jax.random.PRNGKey(seed))
            disp = info.params - center
            if float(jnp.dot(eigvec, disp)) < 0:
                eigvec = -eigvec
            _cache[cache_key] = eigvec

        direction = _cache[cache_key]
        mag = step_size * (basin.radius if scale_by_radius else 1.0)
        return mag * direction
    return walk


def gradient_orthogonal(loss_fn, step_size=0.05, num_iters=10,
                        scale_by_radius=True):
    r"""Project out the gradient and walk along the softest orthogonal mode.

    Avoids re-entry by construction: the walk direction has zero component
    along the steepest-descent direction that pulls back to the basin.

    Cost: *num_iters* HVPs + 1 gradient per basin, cached.
    """
    _cache = {}

    def _hvp(params, v):
        return jax.jvp(jax.grad(loss_fn), (params,), (v,))[1]

    def walk(info, basin):
        cache_key = basin.center.tobytes()
        if cache_key not in _cache:
            center = jnp.array(basin.center)
            n = center.shape[0]
            g = jax.grad(loss_fn)(center)
            g_norm = float(jnp.linalg.norm(g))
            seed = int(np.sum(np.abs(basin.center) * 1e4)) % (2**31)
            key = jax.random.PRNGKey(seed)

            if g_norm > 1e-10:
                g_hat = g / g_norm
                v = jax.random.normal(key, shape=(n,))
                v = v - jnp.dot(v, g_hat) * g_hat
                v = v / jnp.linalg.norm(v)
                for _ in range(num_iters):
                    hv = _hvp(center, v)
                    neg_hv = -hv
                    neg_hv = neg_hv - jnp.dot(neg_hv, g_hat) * g_hat
                    norm = jnp.linalg.norm(neg_hv)
                    v = jnp.where(norm > 1e-30, neg_hv / norm, v)
                direction = v
            else:
                # At true minimum, fall back to unprojected min eigvec
                v = jax.random.normal(key, shape=(n,))
                v = v / jnp.linalg.norm(v)
                for _ in range(num_iters):
                    hv = _hvp(center, v)
                    neg_hv = -hv
                    norm = jnp.linalg.norm(neg_hv)
                    v = jnp.where(norm > 1e-30, neg_hv / norm, v)
                direction = v

            disp = info.params - center
            if float(jnp.dot(direction, disp)) < 0:
                direction = -direction
            _cache[cache_key] = direction

        direction = _cache[cache_key]
        mag = step_size * (basin.radius if scale_by_radius else 1.0)
        return mag * direction
    return walk


def conjugate_basin_escape(loss_fn, step_size=0.05, num_iters=15,
                           max_conjugate=10, scale_by_radius=True):
    r"""Build H-conjugate directions at the basin center; cycle per run_idx.

    Each run exits through a different curvature direction, systematically
    exhausting all escape routes from softest to stiffest.

    Cost: ~(num_iters + max_conjugate) HVPs per basin, cached.
    """
    _cache = {}

    def _hvp(params, v):
        return jax.jvp(jax.grad(loss_fn), (params,), (v,))[1]

    def _build_directions(params, n):
        seed = int(np.sum(np.abs(np.array(params)) * 1e4)) % (2**31)
        key = jax.random.PRNGKey(seed)

        # First direction: minimum eigenvector via power iteration on -H
        v = jax.random.normal(key, shape=(n,))
        v = v / jnp.linalg.norm(v)
        for _ in range(num_iters):
            hv = _hvp(params, v)
            neg_hv = -hv
            norm = jnp.linalg.norm(neg_hv)
            v = jnp.where(norm > 1e-30, neg_hv / norm, v)

        directions = [v]
        num_dirs = min(n, max_conjugate)

        # Lanczos-style: apply H, then orthogonalise
        q = v
        for _ in range(1, num_dirs):
            hq = _hvp(params, q)
            r = hq
            for d_prev in directions:
                r = r - jnp.dot(r, d_prev) * d_prev
            norm = jnp.linalg.norm(r)
            if float(norm) < 1e-12:
                break
            q = r / norm
            directions.append(q)

        return directions

    def walk(info, basin):
        cache_key = basin.center.tobytes()
        if cache_key not in _cache:
            center = jnp.array(basin.center)
            _cache[cache_key] = _build_directions(center, center.shape[0])

        directions = _cache[cache_key]
        idx = info.run_idx % len(directions)
        direction = directions[idx]

        disp = info.params - jnp.array(basin.center)
        if float(jnp.dot(direction, disp)) < 0:
            direction = -direction

        mag = step_size * (basin.radius if scale_by_radius else 1.0)
        return mag * direction
    return walk
