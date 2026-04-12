# optimisers/jax_learnable.py
"""
JAX implementations of induced-metric optimisers with a **learnable diagonal inverse metric**.

This module follows the style of the existing JAX implementations in the repo and
exposes Optax-compatible GradientTransformations.

Provided optimisers:
- custom_sgd_learnable_diag:     Plain embedding f(L)=L (Alg. 1)
- custom_sgd_log_learnable_diag: Log-loss embedding f(L)=log L (Alg. 2; requires passing loss to update())

The learnable inverse metric is parameterised as gamma^{-1}_phi = diag(sigma(s)),
where sigma is controlled by ``metric_param``:

  "exp"              sigma(s) = exp(s)              (default, original)
  "exp_matched_reg"  sigma(s) = exp(s), reg term = reg * exp(s)
  "softplus"         sigma(s) = softplus(s)
  "exp_norm_grad"    sigma(s) = exp(s), s gradient normalised by exp(s)
  "exp_adaptive_clip" sigma(s) = exp(s), clip targets condition number

We update 's' online using a local surrogate objective that maximises
v(phi) = sum(g * (gamma^{-1}_phi g)) with a small LR and light regularisation,
and then mean-centre per leaf to avoid scale degeneracy with xi.
This keeps cost O(N) and preserves SPD by construction.

References to repo paper (algorithms & denominator EMA, momentum, decoupled WD):
- Algorithm 1 & 2 update structure; denominator estimated by short EMA.
"""

from typing import NamedTuple, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import optax

_VALID_METRIC_PARAMS = frozenset({
    "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
})

# Value of s such that softplus(s) = 1  (i.e. s = log(e - 1))
_SOFTPLUS_IDENTITY_INIT = float(jnp.log(jnp.exp(jnp.ones([])) - 1.0))

# ---- utilities ---------------------------------------------------------------

def _tree_dot(a, b):
    """Sum_i a_i * b_i over a pytree of matching structure."""
    return jax.tree.reduce(
        lambda acc, x: acc + jnp.sum(x),
        jax.tree.map(lambda x, y: x * y, a, b),
        initializer=jnp.array(0.0),
    )

def _apply_diag(log_diag, x):
    """Apply diagonal inverse metric diag(exp(log_diag)) to x, per leaf."""
    return jax.tree.map(lambda s, t: jnp.exp(s) * t, log_diag, x)

def _apply_diag_softplus(log_diag, x):
    """Apply diagonal inverse metric diag(softplus(log_diag)) to x, per leaf."""
    return jax.tree.map(lambda s, t: jax.nn.softplus(s) * t, log_diag, x)

def _mean_center(log_diag):
    """Per-leaf mean-centre s to control global scale (keeps gamma^{-1} near identity scale)."""
    return jax.tree.map(lambda s: s - jnp.mean(s), log_diag)

def _clip(log_diag, lo, hi):
    return jax.tree.map(lambda s: jnp.clip(s, lo, hi), log_diag)

def _adaptive_clip(log_diag, max_cond):
    """Clip s values to enforce a maximum condition number across all leaves."""
    all_s = jnp.concatenate([l.ravel() for l in jax.tree.leaves(log_diag)])
    s_median = jnp.median(all_s)
    half_log_cond = 0.5 * jnp.log(max_cond)
    lo = s_median - half_log_cond
    hi = s_median + half_log_cond
    return jax.tree.map(lambda s: jnp.clip(s, lo, hi), log_diag)

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
    metric_param: str = "exp",
    max_condition_number: Optional[float] = None,
) -> optax.GradientTransformation:
    """
    Custom SGD with momentum, decoupled weight decay, *and a learnable diagonal inverse metric*.

    Update matches Algorithm 1 (plain embedding), with:
      r_t = 1 / (1 + |v_hat|),       v = xi * sum_i g_i * (gamma^{-1} g)_i

    The inverse metric is diag(sigma(s)) where sigma is selected by ``metric_param``.
    We update 's' by ascending on v (local surrogate), then apply regularisation,
    per-leaf mean centring, and clip s.

    Args:
        metric_param: Metric parameterization. One of "exp", "exp_matched_reg",
            "softplus", "exp_norm_grad", "exp_adaptive_clip".
        max_condition_number: Target max condition number for adaptive clip mode.
            Only used when metric_param="exp_adaptive_clip". Default 1000.
    """
    if metric_param not in _VALID_METRIC_PARAMS:
        raise ValueError(f"Unknown metric_param={metric_param!r}. "
                         f"Valid: {sorted(_VALID_METRIC_PARAMS)}")

    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)
    use_softplus = metric_param == "softplus"
    use_adaptive_clip = metric_param == "exp_adaptive_clip"
    _max_cond = max_condition_number if max_condition_number is not None else 1000.0

    _apply = _apply_diag_softplus if use_softplus else _apply_diag

    def init(params):
        if use_softplus:
            init_diag = jax.tree.map(
                lambda p: jnp.full_like(p, _SOFTPLUS_IDENTITY_INIT), params)
        else:
            init_diag = jax.tree.map(jnp.zeros_like, params)
        return SGDLearnableDiagState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_diag=init_diag,
        )

    @jax.jit
    def update(grads, state, params=None):
        step = state.step + 1

        # Preconditioned gradient for denominator trace term
        g_tilde = _apply(state.log_diag, grads)
        v = xi * _tree_dot(grads, g_tilde)               # scalar
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = 1.0 / (1.0 + jnp.abs(v_hat))

        # Momentum
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum, grads
        )

        # Apply inverse metric to momentum (with bias correction)
        m_corr = jax.tree.map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)
        m_tilde = _apply(state.log_diag, m_corr)

        # Parameter updates (Optax semantics: 'updates' are added to params)
        if params is None:
            updates = jax.tree.map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree.map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p,
                m_tilde, params
            )

        # --- Online metric learning step (diagonal only) ---
        if metric_param == "softplus":
            # d/ds [xi * softplus(s) * g^2] = xi * softplus(s) * sigmoid(s) * g^2
            s_grad = jax.tree.map(
                lambda s, g: xi * jax.nn.softplus(s) * jax.nn.sigmoid(s) * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        elif metric_param == "exp_matched_reg":
            # Same s_grad, but reg term is reg * exp(s) instead of reg * s
            s_grad = jax.tree.map(
                lambda s, g: xi * jnp.exp(s) * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * jnp.exp(s),
                state.log_diag, s_grad)
        elif metric_param == "exp_norm_grad":
            # Normalise s gradient by exp(s): s_grad = xi * g^2
            s_grad = jax.tree.map(
                lambda s, g: xi * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        else:
            # "exp" and "exp_adaptive_clip": standard s_grad
            s_grad = jax.tree.map(
                lambda s, g: xi * jnp.exp(s) * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)

        new_log_diag = _mean_center(new_log_diag)

        if use_adaptive_clip:
            new_log_diag = _adaptive_clip(new_log_diag, _max_cond)
        else:
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
    metric_param: str = "exp",
    max_condition_number: Optional[float] = None,
) -> optax.GradientTransformation:
    """
    Log-loss embedding version with learnable diagonal inverse metric.

    Update matches Algorithm 2:
      r_t = L_t / (L_t^2 + |v_hat|),   v = xi * sum_i g_i * (gamma^{-1} g)_i

    NOTE: like the repo's log-loss variant, the update function takes 'loss' explicitly:
      updates, state = opt.update(grads, state, loss, params)

    Args:
        metric_param: Metric parameterization. See custom_sgd_learnable_diag docstring.
        max_condition_number: Target max condition number for adaptive clip mode.
    """
    if metric_param not in _VALID_METRIC_PARAMS:
        raise ValueError(f"Unknown metric_param={metric_param!r}. "
                         f"Valid: {sorted(_VALID_METRIC_PARAMS)}")

    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)
    use_softplus = metric_param == "softplus"
    use_adaptive_clip = metric_param == "exp_adaptive_clip"
    _max_cond = max_condition_number if max_condition_number is not None else 1000.0

    _apply = _apply_diag_softplus if use_softplus else _apply_diag

    def init(params):
        if use_softplus:
            init_diag = jax.tree.map(
                lambda p: jnp.full_like(p, _SOFTPLUS_IDENTITY_INIT), params)
        else:
            init_diag = jax.tree.map(jnp.zeros_like, params)
        return SGDLearnableDiagState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_diag=init_diag,
        )

    @jax.jit
    def update(grads, state, loss, params=None):
        step = state.step + 1

        g_tilde = _apply(state.log_diag, grads)
        v = xi * _tree_dot(grads, g_tilde)
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = loss / (jnp.square(loss) + jnp.abs(v_hat))

        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum, grads
        )
        m_corr = jax.tree.map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)
        m_tilde = _apply(state.log_diag, m_corr)

        if params is None:
            updates = jax.tree.map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree.map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p,
                m_tilde, params
            )

        # --- Online metric learning step ---
        if metric_param == "softplus":
            s_grad = jax.tree.map(
                lambda s, g: xi * jax.nn.softplus(s) * jax.nn.sigmoid(s) * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        elif metric_param == "exp_matched_reg":
            s_grad = jax.tree.map(
                lambda s, g: xi * jnp.exp(s) * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * jnp.exp(s),
                state.log_diag, s_grad)
        elif metric_param == "exp_norm_grad":
            s_grad = jax.tree.map(
                lambda s, g: xi * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        else:
            s_grad = jax.tree.map(
                lambda s, g: xi * jnp.exp(s) * (g * g),
                state.log_diag, grads)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)

        new_log_diag = _mean_center(new_log_diag)

        if use_adaptive_clip:
            new_log_diag = _adaptive_clip(new_log_diag, _max_cond)
        else:
            new_log_diag = _clip(new_log_diag, lo, hi)

        new_state = SGDLearnableDiagState(
            step=step,
            momentum=new_momentum,
            metric_ema=new_metric_ema,
            log_diag=new_log_diag,
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)
