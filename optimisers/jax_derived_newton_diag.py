# optimisers/jax_derived_newton_diag.py
"""
JAX implementation of the **derived Newton-targeted diagonal optimizer**.

This optimizer replaces the learnable metric's gradient-ascent s-update
(jax_learnable_diag.py) with a direct EMA towards the Newton target:

    s_{i,t+1} = (1 - beta_s) * s_{i,t} + beta_s * (-log max(H_hat_ii, eps))

where H_hat_ii is the Hutchinson Hessian diagonal estimate.  The derivation
(IMO-83 through IMO-87) shows that this is the unique minimax-optimal diagonal
preconditioner, robust to multiplicative estimation error, and eliminates the
xi/denominator/metric_ema machinery (xi=0, r=1).

Provided optimiser:
- derived_newton_diag:  Plain embedding f(L)=L with Hutchinson-driven metric

The update() signature takes h_diag (Hutchinson Hessian diagonal estimate)
as an extra positional arg, analogous to how log-loss variants take 'loss':

    updates, state = opt.update(grads, state, h_diag, params)

A helper function hutchinson_hvp_diag() is provided for computing h_diag
from a loss function using forward-over-reverse AD (~2x gradient cost).

References:
- IMO-84: Newton target uniqueness theorem
- IMO-85: Robustness theorem (Hutchinson over secant)
- IMO-86: xi=0 optimality
- IMO-87: Plain embedding non-interference
"""

from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
import optax


# ---- utilities (duplicated from jax_learnable_diag.py, matching convention) --

def _apply_diag(log_diag, x):
    """Apply diagonal inverse metric diag(exp(log_diag)) to x, per leaf."""
    return jax.tree.map(lambda s, t: jnp.exp(s) * t, log_diag, x)


def _mean_center(log_diag):
    """Per-leaf mean-centre s to control global scale."""
    return jax.tree.map(lambda s: s - jnp.mean(s), log_diag)


def _clip(log_diag, lo, hi):
    return jax.tree.map(lambda s: jnp.clip(s, lo, hi), log_diag)


# ---- Hutchinson helper ------------------------------------------------------

def _rademacher_like(rng_key, params):
    """Generate a Rademacher (+/-1) pytree matching the structure of params."""
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(rng_key, len(leaves))
    v_leaves = [
        2.0 * jax.random.bernoulli(k, 0.5, l.shape).astype(l.dtype) - 1.0
        for k, l in zip(keys, leaves)
    ]
    return treedef.unflatten(v_leaves)


def hutchinson_hvp_diag(loss_fn, params, rng_key):
    """Estimate Hessian diagonal via Hutchinson's method with Rademacher vectors.

    Uses forward-over-reverse AD: a single ``jax.jvp(jax.grad(loss_fn), ...)``
    call gives both the gradient and the Hessian-vector product.  Total cost is
    approximately 2x a single gradient evaluation.

    Parameters
    ----------
    loss_fn : callable
        Scalar loss function: ``loss_fn(params) -> scalar``.
    params : pytree
        Current parameters.
    rng_key : jax.random.PRNGKey
        PRNG key for Rademacher vector sampling.

    Returns
    -------
    h_diag : pytree
        Estimated Hessian diagonal (same structure as params).
    """
    v = _rademacher_like(rng_key, params)
    _, hvp = jax.jvp(jax.grad(loss_fn), (params,), (v,))
    return jax.tree.map(lambda vi, hi: vi * hi, v, hvp)


# ---- state container --------------------------------------------------------

class DerivedNewtonDiagState(NamedTuple):
    """State for the derived Newton-targeted diagonal optimizer."""
    step: jnp.ndarray       # scalar int32
    momentum: Any            # pytree like params
    log_diag: Any            # pytree like params (the s_i values)


# ---- core builder -----------------------------------------------------------

def derived_newton_diag(
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    beta_s: float = 0.1,
    weight_decay: float = 0.0,
    metric_clip: float = 4.0,
    hess_eps: float = 1e-6,
) -> optax.GradientTransformation:
    """
    Derived Newton-targeted diagonal optimizer.

    The inverse metric diag(exp(s)) is driven towards the Newton target
    s_i* = -log H_ii via an EMA of Hutchinson Hessian diagonal estimates.
    No xi, no denominator, no metric parameterisation variants.

    The update function takes ``h_diag`` (Hutchinson diagonal estimate)
    as a positional argument::

        updates, state = opt.update(grads, state, h_diag, params)

    Use :func:`hutchinson_hvp_diag` to compute ``h_diag`` from a loss function.

    Args:
        learning_rate: Parameter step size eta.
        momentum: Momentum coefficient beta_m.
        beta_s: EMA rate for s towards Newton target.  Replaces metric_lr,
            metric_reg, and xi from the learnable-diag family.
        weight_decay: Decoupled weight decay.
        metric_clip: Symmetric clip bound c for s after mean-centering.
        hess_eps: Floor for max(h_diag, eps) before taking log.
    """
    neg_lr = -learning_rate
    one_minus_momentum = 1.0 - momentum
    one_minus_beta_s = 1.0 - beta_s
    lo, hi = -abs(metric_clip), abs(metric_clip)

    def init(params):
        return DerivedNewtonDiagState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            log_diag=jax.tree.map(jnp.zeros_like, params),
        )

    @jax.jit
    def update(grads, state, h_diag, params=None):
        step = state.step + 1

        # --- Metric EMA towards Newton target ---
        # s_target_i = -log(max(h_diag_i, eps))
        new_log_diag = jax.tree.map(
            lambda s, h: one_minus_beta_s * s
                         + beta_s * (-jnp.log(jnp.maximum(h, hess_eps))),
            state.log_diag, h_diag,
        )
        new_log_diag = _mean_center(new_log_diag)
        new_log_diag = _clip(new_log_diag, lo, hi)

        # --- Momentum with bias correction ---
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum, grads,
        )
        m_corr = jax.tree.map(
            lambda m: m / (1.0 - (momentum ** step)),
            new_momentum,
        )

        # --- Preconditioned step (xi=0 => r=1) ---
        m_tilde = _apply_diag(new_log_diag, m_corr)

        if params is None:
            updates = jax.tree.map(lambda mt: neg_lr * mt, m_tilde)
        else:
            updates = jax.tree.map(
                lambda mt, p: neg_lr * mt - learning_rate * weight_decay * p,
                m_tilde, params,
            )

        new_state = DerivedNewtonDiagState(
            step=step,
            momentum=new_momentum,
            log_diag=new_log_diag,
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)
