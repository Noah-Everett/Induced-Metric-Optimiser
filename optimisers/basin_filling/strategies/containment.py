"""
Containment strategies: determine if a point is inside a known basin.

Each factory returns a callable with signature::

    check(params: ndarray, archive: List[Basin]) -> Optional[int]

Heuristic: sphere_check, loss_gated_check
Geometric: ellipsoid_check
"""

import jax
import jax.numpy as jnp
import numpy as np


def sphere_check():
    """Parameter-space Euclidean sphere around each basin center."""
    def check(params, archive):
        p = np.array(params)
        for i, basin in enumerate(archive):
            if np.linalg.norm(p - basin.center) < basin.radius:
                return i
        return None
    return check


def ellipsoid_check():
    r"""Anisotropic containment using Hessian eigendecomposition.

    Requires basin.extra to contain 'eigenvalues' and 'eigenvectors'
    (populated by the hessian_ellipsoid radius strategy).  Falls back
    to sphere_check if the Hessian data is missing.
    """
    def check(params, archive):
        p = np.array(params)
        for i, basin in enumerate(archive):
            if 'eigenvalues' not in basin.extra:
                # Fallback to sphere
                if np.linalg.norm(p - basin.center) < basin.radius:
                    return i
                continue

            delta = p - basin.center
            eigvals = basin.extra['eigenvalues']
            eigvecs = basin.extra['eigenvectors']  # (k, n)
            semi = basin.extra['semi_axes']

            projections = eigvecs @ delta  # (k,)
            # Normalised ellipsoid: sum (proj_i / semi_i)^2 < 1
            dist_sq = float(np.sum((projections / semi) ** 2))
            if dist_sq < 1.0:
                return i
        return None
    return check


def loss_gated_check(loss_fn, loss_margin=0.1, distance_factor=3.0):
    """Containment by loss sublevel set, gated on loose proximity.

    Combines parameter-space distance (cheap gate) with a loss check
    (rejects false positives from neighbouring minima).
    """
    def check(params, archive):
        p = np.array(params)
        for i, basin in enumerate(archive):
            dist = np.linalg.norm(p - basin.center)
            if dist >= distance_factor * basin.radius:
                continue
            current_loss = float(loss_fn(jnp.array(params)))
            if current_loss < basin.loss + loss_margin:
                return i
        return None
    return check
