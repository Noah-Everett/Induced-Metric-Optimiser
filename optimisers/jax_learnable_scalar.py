# optimisers/jax_scalar_learnable.py
"""
JAX/Optax implementations of induced-metric SGD with a **learnable scalar inverse metric**.

- custom_sgd_learnable_scalar     : Plain embedding (Alg. 1)
- custom_sgd_log_learnable_scalar : Log-loss embedding (Alg. 2; update expects loss)

Inverse metric: gamma^{-1}_phi = sigma(s) * I, where sigma is controlled by
``metric_param`` (see jax_learnable_diag.py for the full list of options).

We learn scalar s online each step by ascending a local surrogate objective
proportional to v = xi * sum(g * (gamma^{-1} g)), regularized and clipped in
log-domain. Momentum, decoupled weight decay, and EMA of the denominator follow
the paper's algorithms (Appendix A code style).

This mirrors the repo's Optax style: return a GradientTransformation with init/update.
"""

from typing import NamedTuple, Optional, Any
import jax
import jax.numpy as jnp
import optax

_VALID_METRIC_PARAMS = frozenset({
    "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
})

_SOFTPLUS_IDENTITY_INIT = float(jnp.log(jnp.exp(jnp.ones([])) - 1.0))

def _tree_dot(a, b):
    return jax.tree.reduce(
        lambda acc, x: acc + jnp.sum(x),
        jax.tree.map(lambda x, y: x * y, a, b),
        initializer=jnp.array(0.0),
    )

def _scale_tree(x, scalar):
    return jax.tree.map(lambda t: scalar * t, x)

class SGDLearnableScalarState(NamedTuple):
    step: jnp.ndarray            # int32 scalar
    momentum: Any                # pytree (like params)
    metric_ema: jnp.ndarray      # scalar
    log_scale: jnp.ndarray       # scalar (s), gamma^{-1} = sigma(s) * I

def custom_sgd_learnable_scalar(
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
    Plain embedding (Alg. 1) with learnable scalar inverse metric.

    r_t = 1 / (1 + |v_hat|),  where v = xi * sigma(s) * sum_i g_i^2

    Args:
        metric_param: Metric parameterization. One of "exp", "exp_matched_reg",
            "softplus", "exp_norm_grad", "exp_adaptive_clip".
            Note: adaptive clip is a no-op for scalar (condition number is always 1).
        max_condition_number: Unused for scalar variant (no anisotropy).
    """
    if metric_param not in _VALID_METRIC_PARAMS:
        raise ValueError(f"Unknown metric_param={metric_param!r}. "
                         f"Valid: {sorted(_VALID_METRIC_PARAMS)}")

    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)
    use_softplus = metric_param == "softplus"

    def _sigma(s):
        return jax.nn.softplus(s) if use_softplus else jnp.exp(s)

    def init(params):
        init_s = jnp.array(_SOFTPLUS_IDENTITY_INIT) if use_softplus else jnp.zeros([])
        return SGDLearnableScalarState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_scale=init_s,
        )

    @jax.jit
    def update(grads, state, params=None):
        step = state.step + 1
        alpha = _sigma(state.log_scale)

        # v = xi * sum g * (alpha g) = xi * alpha * sum g^2
        g_tilde = _scale_tree(grads, alpha)
        v = xi * _tree_dot(grads, g_tilde)
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = 1.0 / (1.0 + jnp.abs(v_hat))

        # momentum (EMA) and bias correction
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g, state.momentum, grads
        )
        m_corr = jax.tree.map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)

        # apply inverse metric (scalar) then step + decoupled WD
        m_tilde = _scale_tree(m_corr, alpha)
        if params is None:
            updates = jax.tree.map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree.map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p, m_tilde, params
            )

        # --- online metric update on s ---
        g_sq_sum = _tree_dot(grads, grads)

        if metric_param == "softplus":
            # d/ds [xi * softplus(s) * sum g^2] = xi * softplus(s) * sigmoid(s) * sum g^2
            grad_s = xi * jax.nn.softplus(state.log_scale) * jax.nn.sigmoid(state.log_scale) * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale
        elif metric_param == "exp_matched_reg":
            grad_s = xi * alpha * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * alpha
        elif metric_param == "exp_norm_grad":
            # Normalise by exp(s): grad_s = xi * g^2
            grad_s = xi * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale
        else:
            # "exp" and "exp_adaptive_clip"
            grad_s = xi * alpha * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale

        # Scalar: adaptive clip is a no-op (condition number = 1), use fixed clip
        new_log_scale = jnp.clip(new_log_scale, lo, hi)

        new_state = SGDLearnableScalarState(
            step=step, momentum=new_momentum, metric_ema=new_metric_ema, log_scale=new_log_scale
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)


def custom_sgd_log_learnable_scalar(
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
    Log-loss embedding (Alg. 2) with learnable scalar inverse metric.

    r_t = L_t / (L_t^2 + |v_hat|),  v = xi * sigma(s) * sum_i g_i^2
    Update signature: updates, state = opt.update(grads, state, loss, params)

    Args:
        metric_param: Metric parameterization. See custom_sgd_learnable_scalar docstring.
        max_condition_number: Unused for scalar variant.
    """
    if metric_param not in _VALID_METRIC_PARAMS:
        raise ValueError(f"Unknown metric_param={metric_param!r}. "
                         f"Valid: {sorted(_VALID_METRIC_PARAMS)}")

    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)
    use_softplus = metric_param == "softplus"

    def _sigma(s):
        return jax.nn.softplus(s) if use_softplus else jnp.exp(s)

    def init(params):
        init_s = jnp.array(_SOFTPLUS_IDENTITY_INIT) if use_softplus else jnp.zeros([])
        return SGDLearnableScalarState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_scale=init_s,
        )

    @jax.jit
    def update(grads, state, loss, params=None):
        step = state.step + 1
        alpha = _sigma(state.log_scale)

        g_tilde = _scale_tree(grads, alpha)
        v = xi * _tree_dot(grads, g_tilde)
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = loss / (jnp.square(loss) + jnp.abs(v_hat))

        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g, state.momentum, grads
        )
        m_corr = jax.tree.map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)

        m_tilde = _scale_tree(m_corr, alpha)
        if params is None:
            updates = jax.tree.map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree.map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p, m_tilde, params
            )

        g_sq_sum = _tree_dot(grads, grads)

        if metric_param == "softplus":
            grad_s = xi * jax.nn.softplus(state.log_scale) * jax.nn.sigmoid(state.log_scale) * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale
        elif metric_param == "exp_matched_reg":
            grad_s = xi * alpha * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * alpha
        elif metric_param == "exp_norm_grad":
            grad_s = xi * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale
        else:
            grad_s = xi * alpha * g_sq_sum
            new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale

        new_log_scale = jnp.clip(new_log_scale, lo, hi)

        new_state = SGDLearnableScalarState(
            step=step, momentum=new_momentum, metric_ema=new_metric_ema, log_scale=new_log_scale
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)
