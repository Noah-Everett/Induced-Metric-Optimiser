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

Aliases supported (case-insensitive):

    * base_mode:
        - grad: "grad", "g", "gr", "gradient", "l"
        - momentum: "momentum", "mom", "m"
    * a_mode / b_mode:
        - zero: "zero", "z", "0", "none"
        - grad: same as base_mode "grad" aliases
        - momentum: same as base_mode "momentum" aliases
        - params: "params", "param", "p", "theta"
        - b_mode only: same_as_a: "same_as_a", "same", "a"

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
    """Optimizer state for custom_sgd_offdiag.

    Hyperparameters are stored as JAX scalars so that the ``update``
    function's traced computation graph is independent of their values.
    This guarantees stable JIT caching across Optuna trials.
    """
    step: jnp.ndarray
    momentum: Any          # Momentum buffer (PyTree, same structure as params)
    zero_buffer: Any       # Pre-allocated zero PyTree (same structure as params)
    neg_lr: jnp.ndarray    # -learning_rate
    mom_coeff: jnp.ndarray # momentum coefficient
    one_minus_mom: jnp.ndarray  # 1 - momentum
    xi: jnp.ndarray
    gamma: jnp.ndarray
    wd_lr: jnp.ndarray    # learning_rate * weight_decay


# -----------------------------------------------
# PyTree-native math helpers
# -----------------------------------------------

def _tree_dot(tree_a, tree_b):
    """Dot product across two PyTrees: sum of all element-wise products."""
    products = jax.tree.map(lambda a, b: jnp.sum(a * b), tree_a, tree_b)
    return jax.tree.reduce(
        lambda acc, x: acc + x, products,
        initializer=jnp.float32(0.0),
    )


def _apply_inverse_rank2(r_tree, l_tree, a_tree, b_tree, gamma, xi):
    r"""Apply :math:`(\\gamma I + \\xi (a l^T + l b^T + l l^T))^{-1} r`
    entirely in PyTree space (rank-2 Woodbury with Cramer's rule).

    Decomposes the metric as :math:`\\gamma I + \\xi U V^T` with
    ``U = [a, l]``, ``V = [l, b+l]`` and solves the resulting 2×2
    system via Cramer's rule — no flattening, no ``linalg.solve``.
    """
    inv_gamma = 1.0 / gamma
    alpha = xi * inv_gamma

    z_tree = jax.tree.map(lambda r: inv_gamma * r, r_tree)
    bl_tree = jax.tree.map(lambda b, l: b + l, b_tree, l_tree)

    # 2×2 matrix M = I + alpha * V^T U
    M00 = 1.0 + alpha * _tree_dot(l_tree, a_tree)
    M01 = alpha * _tree_dot(l_tree, l_tree)
    M10 = alpha * _tree_dot(bl_tree, a_tree)
    M11 = 1.0 + alpha * _tree_dot(bl_tree, l_tree)

    rhs0 = _tree_dot(l_tree, z_tree)
    rhs1 = _tree_dot(bl_tree, z_tree)

    # Cramer's rule for the 2×2 system
    det = M00 * M11 - M01 * M10
    inv_det = 1.0 / det
    w0 = inv_det * (M11 * rhs0 - M01 * rhs1)
    w1 = inv_det * (M00 * rhs1 - M10 * rhs0)

    # y = z - alpha * (a * w0 + l * w1)
    y_tree = jax.tree.map(
        lambda z, a, l: z - alpha * (a * w0 + l * w1),
        z_tree, a_tree, l_tree,
    )
    return y_tree


def _apply_inverse_rank1_a_zero(r_tree, l_tree, b_tree, gamma, xi):
    r"""Sherman-Morrison for ``a = 0``.

    :math:`g_{pb} = \\gamma I + \\xi\\, l\\,(b+l)^T`  (rank-1).

    Inverse via Sherman-Morrison:

    .. math::
        g_{pb}^{-1} r = z - \\alpha\\, l \\;
        \\frac{\\langle b+l,\\, z\\rangle}{1 + \\alpha\\,\\langle b+l,\\, l\\rangle}

    where :math:`z = r/\\gamma`, :math:`\\alpha = \\xi/\\gamma`.
    """
    inv_gamma = 1.0 / gamma
    alpha = xi * inv_gamma

    z_tree = jax.tree.map(lambda r: inv_gamma * r, r_tree)
    bl_tree = jax.tree.map(lambda b, l: b + l, b_tree, l_tree)

    bl_dot_z = _tree_dot(bl_tree, z_tree)
    bl_dot_l = _tree_dot(bl_tree, l_tree)

    scale = alpha * bl_dot_z / (1.0 + alpha * bl_dot_l)

    y_tree = jax.tree.map(lambda z, l: z - scale * l, z_tree, l_tree)
    return y_tree


def _apply_inverse_rank1_b_zero(r_tree, l_tree, a_tree, gamma, xi):
    r"""Sherman-Morrison for ``b = 0``.

    :math:`g_{pb} = \\gamma I + \\xi\\,(a+l)\\, l^T`  (rank-1).

    Inverse via Sherman-Morrison:

    .. math::
        g_{pb}^{-1} r = z - \\alpha\\,(a+l)\\;
        \\frac{\\langle l,\\, z\\rangle}{1 + \\alpha\\,\\langle l,\\, a+l\\rangle}
    """
    inv_gamma = 1.0 / gamma
    alpha = xi * inv_gamma

    z_tree = jax.tree.map(lambda r: inv_gamma * r, r_tree)
    al_tree = jax.tree.map(lambda a, l: a + l, a_tree, l_tree)

    l_dot_z = _tree_dot(l_tree, z_tree)
    l_dot_al = _tree_dot(l_tree, al_tree)

    scale = alpha * l_dot_z / (1.0 + alpha * l_dot_al)

    y_tree = jax.tree.map(lambda z, al: z - scale * al, z_tree, al_tree)
    return y_tree


# -----------------------------------------------
# Mode alias normalization
# -----------------------------------------------

def _normalize_base_mode(mode: str) -> str:
    m = (mode or "").lower()
    if m in {"grad", "g", "gr", "gradient", "l"}:
        return "grad"
    if m in {"momentum", "mom", "m"}:
        return "momentum"
    raise ValueError(
        f"Unknown base_mode: {mode}. Accepted: grad (g, gr, gradient), momentum (mom, m)"
    )


def _normalize_vec_mode(mode: str) -> str:
    m = (mode or "").lower()
    if m in {"zero", "z", "0", "none"}:
        return "zero"
    if m in {"grad", "g", "gr", "gradient", "l"}:
        return "grad"
    if m in {"momentum", "mom", "m"}:
        return "momentum"
    if m in {"params", "param", "p", "theta"}:
        return "params"
    raise ValueError(
        f"Unknown vector mode: {mode}. Accepted: zero (z, 0, none), "
        f"grad (g, gr, gradient), momentum (mom, m), params (param, p, theta)"
    )


def _normalize_b_mode(mode: str) -> str:
    m = (mode or "").lower()
    if m in {"same_as_a", "same", "a"}:
        return "same_as_a"
    try:
        return _normalize_vec_mode(m)
    except ValueError:
        raise ValueError(
            f"Unknown b_mode: {mode}. Accepted: same_as_a (same, a) or any vector mode alias"
        )


# -----------------------------------------------
# Public API
# -----------------------------------------------

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

    # Normalize modes once (not per-step)
    base_mode_norm = _normalize_base_mode(base_mode)
    a_mode_norm = _normalize_vec_mode(a_mode)
    b_mode_norm = _normalize_b_mode(b_mode)

    # Resolve "same_as_a" before we decide the metric rank
    if b_mode_norm == "same_as_a":
        b_mode_resolved = a_mode_norm
    else:
        b_mode_resolved = b_mode_norm

    # Determine metric rank for choosing the optimal inverse formula.
    # Custom a_fn / b_fn / a_static / b_static → always use rank-2.
    has_custom_a = a_fn is not None or a_static is not None
    has_custom_b = b_fn is not None or b_static is not None
    a_is_zero = (not has_custom_a) and a_mode_norm == "zero"
    b_is_zero = (not has_custom_b) and b_mode_resolved == "zero"

    def init(params):
        return OffDiagSGDState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            zero_buffer=jax.tree.map(jnp.zeros_like, params),
            neg_lr=jnp.float32(-learning_rate),
            mom_coeff=jnp.float32(momentum),
            one_minus_mom=jnp.float32(1.0 - momentum),
            xi=jnp.float32(xi),
            gamma=jnp.float32(gamma),
            wd_lr=jnp.float32(learning_rate * weight_decay),
        )

    def update(grads, state, params=None):
        if params is None:
            raise ValueError("params must be provided to custom_sgd_offdiag.update")

        step = state.step + 1
        s_neg_lr = state.neg_lr
        s_mom = state.mom_coeff
        s_one_minus_mom = state.one_minus_mom
        s_xi = state.xi
        s_gamma = state.gamma
        s_wd_lr = state.wd_lr

        # Momentum update
        new_momentum = jax.tree.map(
            lambda m, g: s_mom * m + s_one_minus_mom * g,
            state.momentum,
            grads,
        )

        # Bias-corrected momentum
        mom_correction = 1.0 - s_mom ** step
        m_hat = jax.tree.map(lambda m: m / mom_correction, new_momentum)

        # Choose base vector l
        if base_mode_norm == "grad":
            l_tree = grads
        else:  # "momentum"
            l_tree = m_hat

        # Choose r (vector to be preconditioned)
        r_tree = m_hat if use_momentum_for_update else l_tree

        # ---- build a_tree ----
        if a_fn is not None:
            a_tree = a_fn(params, grads, m_hat, step)
        elif a_static is not None:
            a_tree = a_static
        elif a_mode_norm == "zero":
            a_tree = state.zero_buffer
        elif a_mode_norm == "grad":
            a_tree = grads
        elif a_mode_norm == "momentum":
            a_tree = m_hat
        else:  # "params"
            a_tree = params

        # ---- build b_tree ----
        if b_fn is not None:
            b_tree = b_fn(params, grads, m_hat, step)
        elif b_static is not None:
            b_tree = b_static
        elif b_mode_resolved == "zero":
            b_tree = state.zero_buffer
        elif b_mode_resolved == "grad":
            b_tree = grads
        elif b_mode_resolved == "momentum":
            b_tree = m_hat
        else:  # "params"
            b_tree = params

        # Apply inverse pullback metric (PyTree-native, no flatten/unflatten)
        if a_is_zero:
            # Rank-1 Sherman-Morrison: g_pb = gamma*I + xi*l*(b+l)^T
            y_tree = _apply_inverse_rank1_a_zero(
                r_tree, l_tree, b_tree, s_gamma, s_xi,
            )
        elif b_is_zero:
            # Rank-1 Sherman-Morrison: g_pb = gamma*I + xi*(a+l)*l^T
            y_tree = _apply_inverse_rank1_b_zero(
                r_tree, l_tree, a_tree, s_gamma, s_xi,
            )
        else:
            # Full rank-2 Woodbury with Cramer's rule
            y_tree = _apply_inverse_rank2(
                r_tree, l_tree, a_tree, b_tree, s_gamma, s_xi,
            )

        # delta_theta = -lr * y - lr * weight_decay * theta
        updates = jax.tree.map(
            lambda y, p: s_neg_lr * y - s_wd_lr * p,
            y_tree,
            params,
        )

        new_state = OffDiagSGDState(
            step=step,
            momentum=new_momentum,
            zero_buffer=state.zero_buffer,
            neg_lr=s_neg_lr,
            mom_coeff=s_mom,
            one_minus_mom=s_one_minus_mom,
            xi=s_xi,
            gamma=s_gamma,
            wd_lr=s_wd_lr,
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)
