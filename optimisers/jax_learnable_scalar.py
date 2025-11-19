# optimisers/jax_scalar_learnable.py
"""
JAX/Optax implementations of induced-metric SGD with a **learnable scalar inverse metric**.

- custom_sgd_learnable_scalar     : Plain embedding (Alg. 1)
- custom_sgd_log_learnable_scalar : Log-loss embedding (Alg. 2; update expects loss)

Inverse metric: gamma^{-1}_phi = exp(s) * I, with scalar s learned online each step by
ascending a local surrogate objective proportional to v = xi * sum(g * (gamma^{-1} g)),
regularized and clipped in log-domain. Momentum, decoupled weight decay, and EMA of the
denominator follow the paper’s algorithms (Appendix A code style).

This mirrors the repo’s Optax style: return a GradientTransformation with init/update.
"""

from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
import optax

def _tree_dot(a, b):
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(x),
        jax.tree_util.tree_map(lambda x, y: x * y, a, b),
        initializer=jnp.array(0.0),
    )

def _scale_tree(x, scalar):
    return jax.tree_util.tree_map(lambda t: scalar * t, x)

class SGDLearnableScalarState(NamedTuple):
    step: jnp.ndarray            # int32 scalar
    momentum: Any                # pytree (like params)
    metric_ema: jnp.ndarray      # scalar
    log_scale: jnp.ndarray       # scalar (s), gamma^{-1} = exp(s) * I

def custom_sgd_learnable_scalar(
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
    Plain embedding (Alg. 1) with learnable scalar inverse metric.

    r_t = 1 / (1 + |v_hat|),  where v = xi * sum_i g_i * (gamma^{-1} g)_i = xi * exp(s) * sum_i g_i^2
    Momentum and decoupled WD match the paper’s implementation.
    """
    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)

    def init(params):
        return SGDLearnableScalarState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_scale=jnp.zeros([]),  # s = 0 => gamma^{-1}=I
        )

    @jax.jit
    def update(grads, state, params=None):
        step = state.step + 1
        alpha = jnp.exp(state.log_scale)

        # v = xi * sum g * (alpha g) = xi * alpha * sum g^2
        g_tilde = _scale_tree(grads, alpha)
        v = xi * _tree_dot(grads, g_tilde)
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = 1.0 / (1.0 + jnp.abs(v_hat))

        # momentum (EMA) and bias correction
        new_momentum = jax.tree_util.tree_map(
            lambda m, g: momentum * m + one_minus_momentum * g, state.momentum, grads
        )
        m_corr = jax.tree_util.tree_map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)

        # apply inverse metric (scalar) then step + decoupled WD
        m_tilde = _scale_tree(m_corr, alpha)
        if params is None:
            updates = jax.tree_util.tree_map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree_util.tree_map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p, m_tilde, params
            )

        # online metric update on s: d/ds v = xi * exp(s) * sum g^2  (we use grads from this step)
        # Using s <- s + metric_lr * (d v/ds) - metric_lr * metric_reg * s
        grad_s = xi * alpha * _tree_dot(grads, grads)
        new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale
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
) -> optax.GradientTransformation:
    """
    Log-loss embedding (Alg. 2) with learnable scalar inverse metric.

    r_t = L_t / (L_t^2 + |v_hat|),  v = xi * exp(s) * sum_i g_i^2
    Update signature: updates, state = opt.update(grads, state, loss, params)
    """
    neg_lr = -learning_rate
    one_minus_beta = 1.0 - beta
    one_minus_momentum = 1.0 - momentum
    lo, hi = -abs(metric_clip), abs(metric_clip)

    def init(params):
        return SGDLearnableScalarState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            metric_ema=jnp.zeros([]),
            log_scale=jnp.zeros([]),
        )

    @jax.jit
    def update(grads, state, loss, params=None):
        step = state.step + 1
        alpha = jnp.exp(state.log_scale)

        g_tilde = _scale_tree(grads, alpha)
        v = xi * _tree_dot(grads, g_tilde)
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = loss / (jnp.square(loss) + jnp.abs(v_hat))

        new_momentum = jax.tree_util.tree_map(
            lambda m, g: momentum * m + one_minus_momentum * g, state.momentum, grads
        )
        m_corr = jax.tree_util.tree_map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)

        m_tilde = _scale_tree(m_corr, alpha)
        if params is None:
            updates = jax.tree_util.tree_map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree_util.tree_map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p, m_tilde, params
            )

        grad_s = xi * alpha * _tree_dot(grads, grads)
        new_log_scale = state.log_scale + metric_lr * grad_s - metric_lr * metric_reg * state.log_scale
        new_log_scale = jnp.clip(new_log_scale, lo, hi)

        new_state = SGDLearnableScalarState(
            step=step, momentum=new_momentum, metric_ema=new_metric_ema, log_scale=new_log_scale
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)
