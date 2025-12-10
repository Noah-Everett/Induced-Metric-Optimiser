"""
JAX implementation of SGD with an induced metric that has non-zero
off-diagonal blocks in the ambient metric.

Ambient metric (in (theta, L) space):
    g_ambient = [[gamma * I, a],
                 [b^T,       1]]

Pullback metric on parameter space (using L = L(theta)):
    g_pb = gamma * I + xi (a l^T + l b^T + l l^T),

where:
    - l is some "loss-gradient-like" direction (e.g. gradient or momentum),
    - a, b are additional directions in parameter space.

The optimizer applies:
    delta_theta = -learning_rate * g_pb^{-1} r

for some update direction r (typically bias-corrected momentum).

This file provides:

    custom_sgd_offdiag(...): an optax.GradientTransformation

The metric structure is highly configurable:

    * base_mode: which vector to use as l (default: "grad")
    * a_mode: how to choose a  (default: "momentum")
    * b_mode: how to choose b  (default: "same_as_a")

AND you can override these with:

    * a_static: a PyTree (same structure as params) used as fixed a
    * b_static: a PyTree used as fixed b
    * a_fn: a callable that returns a PyTree for a at each step
    * b_fn: a callable that returns a PyTree for b at each step

Precedence is:

    1. a_fn / b_fn (if not None)
    2. a_static / b_static (if not None)
    3. a_mode / b_mode

Note:
    - a_static and b_static must flatten to the same total size as params.
    - a_fn and b_fn must be JAX-traceable and return PyTrees matching params.
"""

from typing import NamedTuple, Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax


class OffDiagSGDState(NamedTuple):
    """Optimizer state for custom_sgd_offdiag."""
    step: jnp.ndarray
    momentum: Any  # PyTree with same structure as params


def _tree_to_flat(tree):
    """Flatten a PyTree of arrays to a single 1D jnp.ndarray."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return jnp.array([], dtype=jnp.float32)
    flat_leaves = [jnp.ravel(x) for x in leaves]
    return jnp.concatenate(flat_leaves)


def _flat_to_tree(vec, template_tree):
    """Unflatten a 1D vector back into a PyTree with the shapes of template_tree."""
    leaves, treedef = jax.tree_util.tree_flatten(template_tree)
    if not leaves:
        return template_tree
    sizes = [x.size for x in leaves]
    splits = jnp.split(vec, jnp.cumsum(jnp.array(sizes[:-1])))
    new_leaves = [v.reshape(x.shape) for v, x in zip(splits, leaves)]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _apply_inverse_metric_offdiag(
    r_flat: jnp.ndarray,
    l_flat: jnp.ndarray,
    a_flat: jnp.ndarray,
    b_flat: jnp.ndarray,
    gamma: float,
    xi: float,
) -> jnp.ndarray:
    """Apply (gamma I + xi (a l^T + l b^T + l l^T))^{-1} to vector r (flat).

    Using a rank-2 Woodbury update:

        g_pb = gamma I + xi U V^T
        U = [a, l], V = [l, b + l]

    Then:
        g_pb^{-1} = (1/gamma) (I + (xi/gamma) U V^T)^{-1}

    and we implement (I + B V^T)^{-1} via:

        B = (xi/gamma) U
        y = (I + B V^T)^{-1} ( (1/gamma) r )

    which only requires O(d) work plus a 2×2 solve.
    """
    if r_flat.size == 0:
        return r_flat

    inv_gamma = 1.0 / gamma

    # U and V: (d, 2)
    U = jnp.stack([a_flat, l_flat], axis=1)
    V = jnp.stack([l_flat, b_flat + l_flat], axis=1)

    alpha = xi * inv_gamma  # xi/gamma
    B = alpha * U           # (d, 2)

    z = inv_gamma * r_flat  # (d,)

    # 2x2 matrix M = I_2 + V^T B
    M = jnp.eye(2, dtype=r_flat.dtype) + V.T @ B   # (2, 2)

    # Right-hand side: V^T z (2,)
    rhs = V.T @ z

    # Solve M w = rhs
    w = jnp.linalg.solve(M, rhs)  # (2,)

    # Final result: y = z - B @ w
    y = z - B @ w                 # (d,)

    return y


def custom_sgd_offdiag(
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    xi: float = 0.1,
    weight_decay: float = 0.0,
    gamma: float = 1.0,
    base_mode: str = "grad",
    a_mode: str = "momentum",
    b_mode: str = "same_as_a",
    use_momentum_for_update: bool = True,
    a_static: Optional[Any] = None,
    b_static: Optional[Any] = None,
    a_fn: Optional[Callable[[Any, Any, Any, jnp.ndarray], Any]] = None,
    b_fn: Optional[Callable[[Any, Any, Any, jnp.ndarray], Any]] = None,
):
    """Custom SGD with a non-zero off-diagonal induced metric.

    Args:
        learning_rate: Base learning rate.
        momentum: Momentum factor (0 <= momentum < 1).
        xi: Strength of the rank-2 correction in the pullback metric.
        weight_decay: L2 weight decay coefficient (Euclidean).
        gamma: Scalar base metric factor: g_pb = gamma * I + xi( ... ); gamma > 0.
        base_mode: Which vector to use as l in the metric:
            - "grad":      l = gradients
            - "momentum":  l = bias-corrected momentum
        a_mode: How to choose a if neither a_fn nor a_static is provided:
            - "zero":      a = 0
            - "grad":      a = gradients
            - "momentum":  a = bias-corrected momentum
            - "params":    a = parameters
        b_mode: How to choose b if neither b_fn nor b_static is provided:
            - "same_as_a": b = a
            - "zero":      b = 0
            - "grad":      b = gradients
            - "momentum":  b = bias-corrected momentum
            - "params":    b = parameters
        use_momentum_for_update: If True, precondition the bias-corrected
            momentum; otherwise precondition l itself.
        a_static: Optional PyTree (same structure as params) to use as fixed a.
        b_static: Optional PyTree to use as fixed b.
        a_fn: Optional callable a_fn(params, grads, m_hat, step) -> PyTree for a.
        b_fn: Optional callable b_fn(params, grads, m_hat, step) -> PyTree for b.

    Precedence:
        a: a_fn -> a_static -> a_mode
        b: b_fn -> b_static -> b_mode / "same_as_a"
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    neg_lr = -learning_rate
    one_minus_momentum = 1.0 - momentum

    def init(params):
        return OffDiagSGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
        )

    def update(grads, state, params=None):
        if params is None:
            raise ValueError("params must be provided to custom_sgd_offdiag.update")

        step = state.step + 1

        # Momentum update
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum,
            grads,
        )

        # Bias-corrected momentum
        mom_correction = 1.0 - momentum ** step
        m_hat = jax.tree.map(lambda m: m / mom_correction, new_momentum)

        # Choose base vector l
        if base_mode == "grad":
            l_tree = grads
        elif base_mode == "momentum":
            l_tree = m_hat
        else:
            raise ValueError(f"Unknown base_mode: {base_mode}")

        # Choose r (vector to be preconditioned)
        r_tree = m_hat if use_momentum_for_update else l_tree

        # ---- build a_tree ----
        if a_fn is not None:
            # User-provided callable
            a_tree = a_fn(params, grads, m_hat, step)
        elif a_static is not None:
            # User-provided static PyTree
            a_tree = a_static
        else:
            # Mode-based construction
            if a_mode == "zero":
                a_tree = jax.tree.map(jnp.zeros_like, l_tree)
            elif a_mode == "grad":
                a_tree = grads
            elif a_mode == "momentum":
                a_tree = m_hat
            elif a_mode == "params":
                a_tree = params
            else:
                raise ValueError(f"Unknown a_mode: {a_mode}")

        # ---- build b_tree ----
        if b_fn is not None:
            b_tree = b_fn(params, grads, m_hat, step)
        elif b_static is not None:
            b_tree = b_static
        else:
            if b_mode == "same_as_a":
                b_tree = a_tree
            elif b_mode == "zero":
                b_tree = jax.tree.map(jnp.zeros_like, l_tree)
            elif b_mode == "grad":
                b_tree = grads
            elif b_mode == "momentum":
                b_tree = m_hat
            elif b_mode == "params":
                b_tree = params
            else:
                raise ValueError(f"Unknown b_mode: {b_mode}")

        # Flatten everything to global vectors
        l_flat = _tree_to_flat(l_tree)
        r_flat = _tree_to_flat(r_tree)
        a_flat = _tree_to_flat(a_tree)
        b_flat = _tree_to_flat(b_tree)

        # Apply inverse pullback metric in flat space
        y_flat = _apply_inverse_metric_offdiag(
            r_flat=r_flat,
            l_flat=l_flat,
            a_flat=a_flat,
            b_flat=b_flat,
            gamma=gamma,
            xi=xi,
        )

        # Unflatten y_flat to parameter-shaped PyTree
        y_tree = _flat_to_tree(y_flat, params)

        # Parameter updates:
        #   delta_theta = -lr * y - lr * weight_decay * theta
        updates = jax.tree.map(
            lambda y, p: neg_lr * y - learning_rate * weight_decay * p,
            y_tree,
            params,
        )

        new_state = OffDiagSGDState(step=step, momentum=new_momentum)
        return updates, new_state

    return optax.GradientTransformation(init, update)
