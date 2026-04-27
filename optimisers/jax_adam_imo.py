"""
Adam with rank-1 induced-metric correction (Adam-IMO).

This module implements the bolt-on hybrid proposed in
``notes/hessian-and-metric-design-conversation.md`` §12: use Adam's diagonal
preconditioner ``diag(sqrt(v_hat) + eps)`` as the base metric ``gamma`` in the
Sherman-Morrison induced-metric update, giving Adam's per-coordinate
adaptation plus a rank-1 smooth global clip at O(1) overhead per step.

Per-step::

    mu  = b1 * mu + (1 - b1) * g
    nu  = b2 * nu + (1 - b2) * g**2
    m_hat = mu / (1 - b1**t)
    v_hat = nu / (1 - b2**t)
    u   = m_hat / (sqrt(v_hat) + eps)            # Adam direction
    quad = sum_i  g_i**2 / (sqrt(v_hat_i) + eps) # scalar (one extra dot product)
    c   = 1 / (1 + xi * quad)                    # smooth clip in (0, 1]
    updates = -lr * c * u

Note: the rank-1 quadratic uses raw ``g``; the search direction uses the
bias-corrected momentum ``m_hat``. This asymmetry is deliberate and matches
the algorithm in the experiment spec.

Setting ``xi=0`` reduces this exactly to ``optax.adam``.

Also provided is :func:`adam_clip`, a baseline that wraps ``optax.adam`` with
``optax.clip_by_global_norm`` for hard global-norm clipping at threshold
``max_norm``.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class AdamIMOState(NamedTuple):
    """State for :func:`adam_imo`.

    Attributes:
        step: Iteration counter. Starts at 0 from ``init`` and is incremented
            inside ``update`` *before* it is consulted, so the value stored
            in the returned state is the post-update step count (matches
            optax's ``ScaleByAdamState`` convention).
        mu: First-moment EMA. Pytree matching the params structure.
        nu: Second-moment EMA. Pytree matching the params structure.
    """
    step: jnp.ndarray            # scalar int32
    mu: Any                      # pytree of arrays, shape matches params
    nu: Any                      # pytree of arrays, shape matches params


def adam_imo(
    learning_rate=1e-3,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    xi=0.1,
    weight_decay=0.0,
):
    """Adam with smooth rank-1 induced-metric clipping.

    Args:
        learning_rate: Base learning rate. Scalar or optax schedule callable.
        b1: First-moment EMA decay (Adam beta1).
        b2: Second-moment EMA decay (Adam beta2).
        eps: Numerical floor inside ``sqrt(v_hat) + eps``.
        xi: Strength of the rank-1 induced-metric correction. Must be >= 0.
            ``xi = 0`` makes this identical to ``optax.adam``. Larger ``xi``
            shrinks ``c_t`` and therefore each update.
        weight_decay: Decoupled weight decay (AdamW-style). Requires that
            ``params`` be passed to ``optimizer.update``; if it is not, a
            ``ValueError`` is raised. Default 0 disables.

    Returns:
        optax.GradientTransformation.

    Notes:
        The clip factor is ``c_t = 1 / (1 + xi * sum_i g_i^2 / (sqrt(v_hat_i)
        + eps))``. It is in (0, 1] for any ``xi >= 0`` and decreases
        monotonically in ``xi``.

        The classical induced-metric step bound ``||delta theta|| <= eta /
        (2 sqrt(xi))`` from the Harvey paper holds when ``u_t`` and the
        rank-1 quadratic share the same denominator (e.g. ``gamma = I``,
        where both reduce to the same ``g``). Here ``u_t = m_hat / (sqrt(v_hat)
        + eps)`` uses bias-corrected momentum while ``quad`` uses the raw
        gradient, so ``u_t`` and ``quad`` involve different powers of
        ``(sqrt(v_hat) + eps)``: the AM-GM step in the original derivation
        no longer applies. The optimizer still performs smooth global
        clipping (update norm shrinks monotonically in ``xi`` and goes to
        zero as ``xi * quad -> inf``), but does not inherit the tight
        ``1/(2 sqrt(xi))`` bound.
    """
    if learning_rate is None:
        raise ValueError("learning_rate must be provided")
    if xi < 0:
        raise ValueError(f"xi must be non-negative, got {xi}")

    def _lr(step):
        # Support both scalar and schedule (callable) learning rates.
        return learning_rate(step) if callable(learning_rate) else learning_rate

    def init(params):
        return AdamIMOState(
            step=jnp.zeros([], dtype=jnp.int32),
            mu=jax.tree.map(jnp.zeros_like, params),
            nu=jax.tree.map(jnp.zeros_like, params),
        )

    @jax.jit
    def update(grads, state, params=None):
        step = state.step + 1
        mu = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, grads)
        nu = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state.nu, grads)

        # Bias correction factors
        bc1 = 1.0 - jnp.power(b1, step)
        bc2 = 1.0 - jnp.power(b2, step)

        # Adam direction: m_hat / (sqrt(v_hat) + eps)
        u = jax.tree.map(
            lambda m, v: (m / bc1) / (jnp.sqrt(v / bc2) + eps),
            mu,
            nu,
        )

        # Quadratic form using raw g (not m_hat) and gamma^{-1} = 1/(sqrt(v_hat)+eps)
        per_tensor = jax.tree.map(
            lambda g, v: jnp.sum((g * g) / (jnp.sqrt(v / bc2) + eps)),
            grads,
            nu,
        )
        quad = jax.tree_util.tree_reduce(jnp.add, per_tensor, jnp.array(0.0))
        c = 1.0 / (1.0 + xi * quad)

        lr = _lr(step)
        scale = -lr * c

        if weight_decay != 0.0:
            if params is None:
                raise ValueError(
                    "adam_imo with weight_decay != 0 requires params to be "
                    "passed to optimizer.update(grads, state, params)."
                )
            updates = jax.tree.map(
                lambda u_, p: scale * u_ - lr * weight_decay * p,
                u,
                params,
            )
        else:
            updates = jax.tree.map(lambda u_: scale * u_, u)

        new_state = AdamIMOState(step=step, mu=mu, nu=nu)
        return updates, new_state

    return optax.GradientTransformation(init, update)


def adam_imo_diagnostics(grads, state, xi, eps=1e-8, b2=0.999):
    """Recompute per-step Adam-IMO scalars for logging.

    Used by ``optimizer_diagnostics.collect_diagnostics`` to surface the
    clipping activity ``zeta = xi * quad`` and the resulting clip factor
    ``c_t``. Pure function — does not mutate state.

    Args:
        grads: Current gradients (pytree).
        state: Current :class:`AdamIMOState`. ``state.step`` is the
            **post-update** count (matching optax convention) — i.e. the
            value found in ``opt_state`` after one or more
            ``optimizer.update`` calls. ``b2 ** state.step`` is therefore
            the bias-correction denominator ``bc2`` used by the most-recent
            update. Caller must pass the same ``xi``, ``eps``, ``b2`` that
            were passed to :func:`adam_imo` for the numbers to match what
            the optimizer actually applied.
        xi: Same ``xi`` passed to :func:`adam_imo`.
        eps: Same ``eps`` passed to :func:`adam_imo`.
        b2: Same ``b2`` passed to :func:`adam_imo`.

    Returns:
        Dict with ``quad``, ``zeta``, ``c_t`` as Python floats.
    """
    step = jnp.maximum(state.step, jnp.array(1, dtype=jnp.int32))
    bc2 = 1.0 - jnp.power(b2, step)
    per_tensor = jax.tree.map(
        lambda g, v: jnp.sum((g * g) / (jnp.sqrt(v / bc2) + eps)),
        grads,
        state.nu,
    )
    quad = jax.tree_util.tree_reduce(jnp.add, per_tensor, jnp.array(0.0))
    zeta = xi * quad
    c_t = 1.0 / (1.0 + zeta)
    return {
        "quad": float(quad),
        "zeta": float(zeta),
        "c_t": float(c_t),
    }


def adam_clip(
    learning_rate=1e-3,
    max_norm=1.0,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    weight_decay=0.0,
):
    """Adam with hard global-norm gradient clipping.

    Baseline for the bolt-on experiment: contrasts the smooth Adam-IMO
    clip against the standard hard global-norm clip used by most
    practitioners. Implemented as ``optax.chain(clip_by_global_norm,
    adam[w])``.
    """
    inner = (
        optax.adamw(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps,
                    weight_decay=weight_decay)
        if weight_decay != 0.0
        else optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    )
    return optax.chain(optax.clip_by_global_norm(max_norm), inner)
