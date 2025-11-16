# optimisers/jax_learnable.py
"""
JAX implementations of induced-metric optimisers with a **learnable diagonal inverse metric**.

This module follows the style of the existing JAX implementations in the repo and
exposes Optax-compatible GradientTransformations.

Provided optimisers:
- custom_sgd_learnable_diag:     Plain embedding f(L)=L (Alg. 1)
- custom_sgd_log_learnable_diag: Log-loss embedding f(L)=log L (Alg. 2; requires passing loss to update())

The learnable inverse metric is parameterised as gamma^{-1}_phi = diag(exp(s)),
with one 's' tensor per parameter leaf. We update 's' online using a local surrogate
objective that maximises v(phi) = sum(g * (gamma^{-1}_phi g)) with a small LR and
light regularisation, and then mean-centre per leaf to avoid scale degeneracy with xi.
This keeps cost O(N) and preserves SPD by construction.

References to repo paper (algorithms & denominator EMA, momentum, decoupled WD):
- Algorithm 1 & 2 update structure; denominator estimated by short EMA. 
"""

from typing import NamedTuple, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import optax

# ---- utilities ---------------------------------------------------------------

def _tree_dot(a, b):
    """Sum_i a_i * b_i over a pytree of matching structure."""
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(x),
        jax.tree_util.tree_map(lambda x, y: x * y, a, b),
        initializer=jnp.array(0.0),
    )

def _apply_diag(log_diag, x):
    """Apply diagonal inverse metric diag(exp(log_diag)) to x, per leaf."""
    return jax.tree_util.tree_map(lambda s, t: jnp.exp(s) * t, log_diag, x)

def _mean_center(log_diag):
    """Per-leaf mean-centre s to control global scale (keeps gamma^{-1} near identity scale)."""
    return jax.tree_util.tree_map(lambda s: s - jnp.mean(s), log_diag)

def _clip(log_diag, lo, hi):
    return jax.tree_util.tree_map(lambda s: jnp.clip(s, lo, hi), log_diag)

# ---- state containers --------------------------------------------------------

class SGDLearnableDiagState(NamedTuple):
    """State for SGD with learnable diagonal inverse metric."""
    step: jnp.ndarray                      # scalar int32
    momentum: Any                          # pytree like params
    metric_ema: jnp.ndarray                # scalar
    log_diag: Any                          # pytree like params (same structure as params)

# ---- core builders -----------------------------------------------------------

def custom_sgd_learnable_diag(
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    xi: float = 0.1,
    beta: float = 0.8,
    weight_decay: float = 0.0,
    metric_lr: float = 1e-3,
    metric_reg: float = 1e-4,
    metric_clip: float = 4.0,
) -> optax.GradientTransformation:
    """
    Custom SGD with momentum, decoupled weight decay, *and a learnable diagonal inverse metric*.

    Update matches Algorithm 1 (plain embedding), with:
      r_t = 1 / (1 + |v_hat|),       v = xi * sum_i g_i * (gamma^{-1} g)_i

    The inverse metric is diag(exp(s)); we update 's' by ascending on v (local surrogate),
    then apply L2 regularisation and per-leaf mean centring, and clip s to [-metric_clip, +metric_clip].
    """

    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)

    def init(params):
        return SGDLearnableDiagState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_diag=jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params),
        )

    @jax.jit
    def update(grads, state, params=None):
        step = state.step + 1

        # Preconditioned gradient for denominator trace term
        g_tilde = _apply_diag(state.log_diag, grads)
        v = xi * _tree_dot(grads, g_tilde)               # scalar
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = 1.0 / (1.0 + jnp.abs(v_hat))

        # Momentum
        new_momentum = jax.tree_util.tree_map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum, grads
        )

        # Apply inverse metric to momentum (with bias correction)
        m_corr = jax.tree_util.tree_map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)
        m_tilde = _apply_diag(state.log_diag, m_corr)

        # Parameter updates (Optax semantics: 'updates' are added to params)
        if params is None:
            updates = jax.tree_util.tree_map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree_util.tree_map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p,
                m_tilde, params
            )

        # --- Online metric learning step (diagonal only) ---
        # Exact gradient for s: d/ds sum(exp(s) * g^2) = exp(s) * g^2
        s_grad = jax.tree_util.tree_map(lambda s, g: jnp.exp(s) * (g * g), state.log_diag, grads)
        new_log_diag = jax.tree_util.tree_map(
            lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
            state.log_diag, s_grad
        )
        new_log_diag = _mean_center(new_log_diag)
        new_log_diag = _clip(new_log_diag, lo, hi)

        new_state = SGDLearnableDiagState(
            step=step,
            momentum=new_momentum,
            metric_ema=new_metric_ema,
            log_diag=new_log_diag,
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)


def custom_sgd_log_learnable_diag(
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    xi: float = 0.1,
    beta: float = 0.8,
    weight_decay: float = 0.0,
    metric_lr: float = 1e-3,
    metric_reg: float = 1e-4,
    metric_clip: float = 4.0,
) -> optax.GradientTransformation:
    """
    Log-loss embedding version with learnable diagonal inverse metric.

    Update matches Algorithm 2:
      r_t = L_t / (L_t^2 + |v_hat|),   v = xi * sum_i g_i * (gamma^{-1} g)_i

    NOTE: like the repo's log-loss variant, the update function takes 'loss' explicitly:
      updates, state = opt.update(grads, state, loss, params)
    """

    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)

    def init(params):
        return SGDLearnableDiagState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_diag=jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params),
        )

    @jax.jit
    def update(grads, state, loss, params=None):
        step = state.step + 1

        g_tilde = _apply_diag(state.log_diag, grads)
        v = xi * _tree_dot(grads, g_tilde)
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = loss / (jnp.square(loss) + jnp.abs(v_hat))

        new_momentum = jax.tree_util.tree_map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum, grads
        )
        m_corr = jax.tree_util.tree_map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)
        m_tilde = _apply_diag(state.log_diag, m_corr)

        if params is None:
            updates = jax.tree_util.tree_map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree_util.tree_map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p,
                m_tilde, params
            )

        # Metric learning step (same as plain)
        s_grad = jax.tree_util.tree_map(lambda s, g: jnp.exp(s) * (g * g), state.log_diag, grads)
        new_log_diag = jax.tree_util.tree_map(
            lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
            state.log_diag, s_grad
        )
        new_log_diag = _mean_center(new_log_diag)
        new_log_diag = _clip(new_log_diag, lo, hi)

        new_state = SGDLearnableDiagState(
            step=step,
            momentum=new_momentum,
            metric_ema=new_metric_ema,
            log_diag=new_log_diag,
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)
