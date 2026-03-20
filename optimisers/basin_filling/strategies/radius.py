"""
Radius estimation strategies: determine the extent of a discovered basin.

Each factory returns a callable with signature::

    estimate(info: StepInfo) -> float

Heuristic: fixed, trajectory, loss_contour
Geometric: hessian_ellipsoid, adaptive
"""

import jax
import jax.numpy as jnp
import numpy as np


# ===================================================================
# Heuristic
# ===================================================================

def fixed(value=0.5):
    """Constant radius for all basins."""
    def estimate(info):
        return value
    return estimate


def trajectory(scale=1.0, min_radius=0.01, fallback=0.5):
    """Max distance from stuck point to any earlier trajectory point."""
    def estimate(info):
        normal_pts = [t.params for t in info.trajectory if not t.walkthrough]
        if len(normal_pts) < 2:
            return fallback
        stuck = np.array(info.params)
        dists = np.array([np.linalg.norm(p - stuck) for p in normal_pts])
        return max(float(np.max(dists)) * scale, min_radius)
    return estimate


def loss_contour(factor=0.1, scale=1.0, min_radius=0.01, fallback=0.5):
    """Basin edge at a fraction of the total loss drop along the trajectory."""
    def estimate(info):
        normal = [(t.params, t.loss) for t in info.trajectory
                  if not t.walkthrough]
        if len(normal) < 2:
            return fallback
        stuck = np.array(info.params)
        losses = np.array([l for _, l in normal])
        stuck_loss = losses[-1]
        loss_range = losses[0] - stuck_loss
        if loss_range <= 0:
            return fallback
        contour = stuck_loss + factor * loss_range
        r = min_radius
        for params_i, loss_i in reversed(normal):
            if loss_i > contour:
                r = float(np.linalg.norm(params_i - stuck))
                break
        return max(r * scale, min_radius)
    return estimate


# ===================================================================
# Geometric
# ===================================================================

def hessian_ellipsoid(loss_fn, loss_budget=None, min_radius=0.01,
                      fallback=0.5):
    r"""Radius from Hessian eigenvalues at the minimum.

    In the quadratic approximation L ~ L* + 1/2 d^T H d, the basin
    at loss budget *c* is an ellipsoid with semi-axes sqrt(2c / lam_i).
    Returns the geometric mean of the semi-axes.

    Stores eigenvalues and eigenvectors in the StepInfo for the core
    loop to propagate to Basin.extra (enables ellipsoid containment).

    For 2-D this computes the exact 2x2 Hessian.  For larger problems
    use Lanczos (future extension).
    """
    def estimate(info):
        params = jnp.array(info.params)
        H = jax.hessian(loss_fn)(params)
        H = np.array(H)

        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, 1e-8)  # clamp negative/tiny

        # Determine loss budget
        if loss_budget is not None:
            c = loss_budget
        else:
            normal = [t.loss for t in info.trajectory if not t.walkthrough]
            if len(normal) >= 2:
                c = max(0.1 * (normal[0] - normal[-1]), 1e-8)
            else:
                return fallback

        semi_axes = np.sqrt(2.0 * c / eigvals)
        r = float(np.exp(np.mean(np.log(semi_axes))))

        # Store for ellipsoid containment check
        info._hessian_extra = {
            'eigenvalues': eigvals,
            'eigenvectors': eigvecs.T,  # (k, n)
            'semi_axes': semi_axes,
            'loss_budget': float(c),
        }
        return max(r, min_radius)
    return estimate


def adaptive(initial_radius=0.1, growth_factor=1.5, merge_threshold=0.5,
             min_radius=0.01, max_radius=10.0):
    """Self-correcting radius via convergence feedback across runs.

    Starts conservative and expands when later runs reconverge to the
    same basin.  The core loop calls ``post_run_hook(archive, trajectory)``
    after each run if the attribute exists.
    """
    def estimate(info):
        return initial_radius

    def post_run_hook(archive, trajectory):
        if len(archive) < 2:
            return
        new_basin = archive[-1]
        new_center = np.array(new_basin.center)

        for existing in archive[:-1]:
            dist = np.linalg.norm(new_center - existing.center)
            if dist < (1.0 + merge_threshold) * existing.radius:
                # Reconvergence: expand existing, remove duplicate
                normal_pts = [t.params for t in trajectory
                              if not t.walkthrough]
                if normal_pts:
                    dists = [np.linalg.norm(p - existing.center)
                             for p in normal_pts]
                    farthest = max(dists)
                    new_r = min(max(farthest, existing.radius),
                                existing.radius * growth_factor,
                                max_radius)
                    existing.radius = max(new_r, min_radius)
                archive.pop()
                return

    estimate.post_run_hook = post_run_hook
    return estimate
