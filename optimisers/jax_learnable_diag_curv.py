# optimisers/jax_learnable_diag_curv.py
"""
JAX implementations of induced-metric optimisers with a **curvature-aware learnable diagonal
inverse metric**.

Extends the diagonal learnable variant (jax_learnable_diag.py) with a secant-based Hessian
diagonal estimator that drives anti-correlation between metric scale and curvature sign.
Setting curv_beta=0 recovers the original diagonal learnable rule exactly.

Provided optimisers:
- custom_sgd_learnable_diag_curv:     Plain embedding f(L)=L (Alg. 1 + curvature correction)
- custom_sgd_log_learnable_diag_curv: Log-loss embedding f(L)=log L (Alg. 2 + curvature correction)

The curvature-aware metric update is:
  s_i <- s_i + mu * [xi * sigma(s_i) * g_i^2 * dsigma
                      - curv_beta * tanh(H_hat_ii / curv_tau) * sigma(s_i) * g_i^2 * dsigma]
                - mu * reg_term(s_i)

where sigma and dsigma depend on metric_param (see jax_learnable_diag.py),
and H_hat_ii is estimated via secant differences:
  H_hat_ii ~ (g_i^{(t)} - g_i^{(t-1)}) / (theta_i^{(t)} - theta_i^{(t-1)})

Reference: Phiacta entry "Optimal Curvature Correction and Proposed Update Rule"
           (1998d606-92de-4a33-9345-bcf7809f9a6c)
"""

from typing import NamedTuple, Optional, Any
import jax
import jax.numpy as jnp
import optax

_VALID_METRIC_PARAMS = frozenset({
    "exp", "exp_matched_reg", "softplus", "exp_norm_grad", "exp_adaptive_clip",
})

_SOFTPLUS_IDENTITY_INIT = float(jnp.log(jnp.exp(jnp.ones([])) - 1.0))

# ---- utilities ---------------------------------------------------------------

from optimisers.jax_diag_utils import (
    apply_diag as _apply_diag,
    apply_diag_softplus as _apply_diag_softplus,
    mean_center as _mean_center,
    clip as _clip,
)

def _tree_dot(a, b):
    """Sum_i a_i * b_i over a pytree of matching structure."""
    return jax.tree.reduce(
        lambda acc, x: acc + jnp.sum(x),
        jax.tree.map(lambda x, y: x * y, a, b),
        initializer=jnp.array(0.0),
    )

def _adaptive_clip(log_diag, max_cond):
    """Clip s values to enforce a maximum condition number across all leaves."""
    all_s = jnp.concatenate([l.ravel() for l in jax.tree.leaves(log_diag)])
    s_median = jnp.median(all_s)
    half_log_cond = 0.5 * jnp.log(max_cond)
    lo = s_median - half_log_cond
    hi = s_median + half_log_cond
    return jax.tree.map(lambda s: jnp.clip(s, lo, hi), log_diag)

# ---- state containers --------------------------------------------------------

class SGDLearnableDiagCurvState(NamedTuple):
    """State for SGD with curvature-aware learnable diagonal inverse metric."""
    step: jnp.ndarray                      # scalar int32
    momentum: Any                          # pytree like params
    metric_ema: jnp.ndarray                # scalar
    log_diag: Any                          # pytree like params
    prev_grads: Any                        # pytree like params (None-tree on step 0)
    prev_params: Any                       # pytree like params (None-tree on step 0)

# ---- core builders -----------------------------------------------------------

def custom_sgd_learnable_diag_curv(
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    xi: float = 0.1,
    beta: float = 0.8,
    weight_decay: float = 0.0,
    metric_lr: float = 1e-3,
    metric_reg: float = 1e-4,
    metric_clip: float = 4.0,
    curv_beta: float = 0.05,
    curv_tau: float = 1.0,
    secant_eps: float = 1e-5,
    metric_param: str = "exp",
    max_condition_number: Optional[float] = None,
) -> optax.GradientTransformation:
    """
    Custom SGD with curvature-aware learnable diagonal inverse metric.

    Extends the diagonal learnable variant with a curvature-sign correction term
    that drives anti-correlation between metric scale and Hessian sign.

    Update matches Algorithm 1 (plain embedding).

    Args:
        learning_rate: Parameter step size eta.
        momentum: Momentum coefficient.
        xi: Loss-dimension weight.
        beta: EMA coefficient for denominator smoothing.
        weight_decay: Decoupled weight decay.
        metric_lr: Metric learning rate mu.
        metric_reg: Metric regularisation lambda_s.
        metric_clip: Bounds on s_i in log-domain.
        curv_beta: Curvature correction strength. 0 recovers the base rule.
        curv_tau: Smoothing temperature for tanh(H_hat/tau).
        secant_eps: Minimum |delta_theta| for secant estimate stability.
        metric_param: Metric parameterization. One of "exp", "exp_matched_reg",
            "softplus", "exp_norm_grad", "exp_adaptive_clip".
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
        zeros = jax.tree.map(jnp.zeros_like, params)
        if use_softplus:
            init_diag = jax.tree.map(
                lambda p: jnp.full_like(p, _SOFTPLUS_IDENTITY_INIT), params)
        else:
            init_diag = jax.tree.map(jnp.zeros_like, params)
        return SGDLearnableDiagCurvState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=zeros,
            metric_ema=jnp.zeros([]),
            log_diag=init_diag,
            prev_grads=zeros,
            prev_params=zeros,
        )

    @jax.jit
    def update(grads, state, params=None):
        step = state.step + 1

        # Preconditioned gradient for denominator trace term
        g_tilde = _apply(state.log_diag, grads)
        v = xi * _tree_dot(grads, g_tilde)
        new_metric_ema = beta * state.metric_ema + one_minus_beta * v
        v_hat = new_metric_ema / (1.0 - (beta ** step))
        r = 1.0 / (1.0 + jnp.abs(v_hat))

        # Momentum
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum, grads
        )

        # Apply inverse metric to bias-corrected momentum
        m_corr = jax.tree.map(lambda m: m / (1.0 - (momentum ** step)), new_momentum)
        m_tilde = _apply(state.log_diag, m_corr)

        # Parameter updates
        if params is None:
            updates = jax.tree.map(lambda mt: neg_lr * r * mt, m_tilde)
        else:
            updates = jax.tree.map(
                lambda mt, p: neg_lr * r * mt - learning_rate * weight_decay * p,
                m_tilde, params
            )

        # --- Online metric learning step (curvature-aware) ---

        # Secant Hessian diagonal estimate
        have_prev = step > 1

        if use_softplus:
            def _secant_curv_correction(s, g, prev_g, p, prev_p):
                delta_theta = p - prev_p
                delta_grad = g - prev_g
                safe = jnp.abs(delta_theta) > secant_eps
                denom = jnp.where(safe, delta_theta, jnp.ones_like(delta_theta))
                h_hat = jnp.where(safe, delta_grad / denom, jnp.zeros_like(g))
                curv_sign = jnp.tanh(h_hat / curv_tau)
                # Curvature correction with softplus chain rule
                return curv_beta * curv_sign * jax.nn.softplus(s) * jax.nn.sigmoid(s) * (g * g)
        elif metric_param == "exp_norm_grad":
            def _secant_curv_correction(s, g, prev_g, p, prev_p):
                delta_theta = p - prev_p
                delta_grad = g - prev_g
                safe = jnp.abs(delta_theta) > secant_eps
                denom = jnp.where(safe, delta_theta, jnp.ones_like(delta_theta))
                h_hat = jnp.where(safe, delta_grad / denom, jnp.zeros_like(g))
                curv_sign = jnp.tanh(h_hat / curv_tau)
                # Normalised: exp(s) cancels out
                return curv_beta * curv_sign * (g * g)
        else:
            def _secant_curv_correction(s, g, prev_g, p, prev_p):
                delta_theta = p - prev_p
                delta_grad = g - prev_g
                safe = jnp.abs(delta_theta) > secant_eps
                denom = jnp.where(safe, delta_theta, jnp.ones_like(delta_theta))
                h_hat = jnp.where(safe, delta_grad / denom, jnp.zeros_like(g))
                curv_sign = jnp.tanh(h_hat / curv_tau)
                return curv_beta * curv_sign * jnp.exp(s) * (g * g)

        if params is not None:
            curv_corr = jax.tree.map(
                _secant_curv_correction,
                state.log_diag, grads, state.prev_grads, params, state.prev_params
            )
            curv_corr = jax.tree.map(lambda c: jnp.where(have_prev, c, 0.0), curv_corr)
        else:
            curv_corr = jax.tree.map(jnp.zeros_like, grads)

        # s_grad = base_drive - curvature_correction
        if metric_param == "softplus":
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * jax.nn.softplus(s) * jax.nn.sigmoid(s) * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        elif metric_param == "exp_matched_reg":
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * jnp.exp(s) * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * jnp.exp(s),
                state.log_diag, s_grad)
        elif metric_param == "exp_norm_grad":
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        else:
            # "exp" and "exp_adaptive_clip"
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * jnp.exp(s) * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)

        new_log_diag = _mean_center(new_log_diag)

        if use_adaptive_clip:
            new_log_diag = _adaptive_clip(new_log_diag, _max_cond)
        else:
            new_log_diag = _clip(new_log_diag, lo, hi)

        # Store current grads and params for next step's secant estimate
        new_prev_grads = grads
        new_prev_params = params if params is not None else jax.tree.map(jnp.zeros_like, grads)

        new_state = SGDLearnableDiagCurvState(
            step=step,
            momentum=new_momentum,
            metric_ema=new_metric_ema,
            log_diag=new_log_diag,
            prev_grads=new_prev_grads,
            prev_params=new_prev_params,
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)


def custom_sgd_log_learnable_diag_curv(
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    xi: float = 0.1,
    beta: float = 0.8,
    weight_decay: float = 0.0,
    metric_lr: float = 1e-3,
    metric_reg: float = 1e-4,
    metric_clip: float = 4.0,
    curv_beta: float = 0.05,
    curv_tau: float = 1.0,
    secant_eps: float = 1e-5,
    metric_param: str = "exp",
    max_condition_number: Optional[float] = None,
) -> optax.GradientTransformation:
    """
    Log-loss embedding version with curvature-aware learnable diagonal inverse metric.

    Update matches Algorithm 2:
      r_t = L_t / (L_t^2 + |v_hat|)

    NOTE: the update function takes 'loss' explicitly:
      updates, state = opt.update(grads, state, loss, params)

    Args:
        metric_param: Metric parameterization. See custom_sgd_learnable_diag_curv docstring.
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
        zeros = jax.tree.map(jnp.zeros_like, params)
        if use_softplus:
            init_diag = jax.tree.map(
                lambda p: jnp.full_like(p, _SOFTPLUS_IDENTITY_INIT), params)
        else:
            init_diag = jax.tree.map(jnp.zeros_like, params)
        return SGDLearnableDiagCurvState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=zeros,
            metric_ema=jnp.zeros([]),
            log_diag=init_diag,
            prev_grads=zeros,
            prev_params=zeros,
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

        # --- Online metric learning step (curvature-aware) ---
        have_prev = step > 1

        if use_softplus:
            def _secant_curv_correction(s, g, prev_g, p, prev_p):
                delta_theta = p - prev_p
                delta_grad = g - prev_g
                safe = jnp.abs(delta_theta) > secant_eps
                denom = jnp.where(safe, delta_theta, jnp.ones_like(delta_theta))
                h_hat = jnp.where(safe, delta_grad / denom, jnp.zeros_like(g))
                curv_sign = jnp.tanh(h_hat / curv_tau)
                return curv_beta * curv_sign * jax.nn.softplus(s) * jax.nn.sigmoid(s) * (g * g)
        elif metric_param == "exp_norm_grad":
            def _secant_curv_correction(s, g, prev_g, p, prev_p):
                delta_theta = p - prev_p
                delta_grad = g - prev_g
                safe = jnp.abs(delta_theta) > secant_eps
                denom = jnp.where(safe, delta_theta, jnp.ones_like(delta_theta))
                h_hat = jnp.where(safe, delta_grad / denom, jnp.zeros_like(g))
                curv_sign = jnp.tanh(h_hat / curv_tau)
                return curv_beta * curv_sign * (g * g)
        else:
            def _secant_curv_correction(s, g, prev_g, p, prev_p):
                delta_theta = p - prev_p
                delta_grad = g - prev_g
                safe = jnp.abs(delta_theta) > secant_eps
                denom = jnp.where(safe, delta_theta, jnp.ones_like(delta_theta))
                h_hat = jnp.where(safe, delta_grad / denom, jnp.zeros_like(g))
                curv_sign = jnp.tanh(h_hat / curv_tau)
                return curv_beta * curv_sign * jnp.exp(s) * (g * g)

        if params is not None:
            curv_corr = jax.tree.map(
                _secant_curv_correction,
                state.log_diag, grads, state.prev_grads, params, state.prev_params
            )
            curv_corr = jax.tree.map(lambda c: jnp.where(have_prev, c, 0.0), curv_corr)
        else:
            curv_corr = jax.tree.map(jnp.zeros_like, grads)

        if metric_param == "softplus":
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * jax.nn.softplus(s) * jax.nn.sigmoid(s) * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        elif metric_param == "exp_matched_reg":
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * jnp.exp(s) * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * jnp.exp(s),
                state.log_diag, s_grad)
        elif metric_param == "exp_norm_grad":
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)
        else:
            s_grad = jax.tree.map(
                lambda s, g, cc: xi * jnp.exp(s) * (g * g) - cc,
                state.log_diag, grads, curv_corr)
            new_log_diag = jax.tree.map(
                lambda s, sg: s + metric_lr * sg - metric_lr * metric_reg * s,
                state.log_diag, s_grad)

        new_log_diag = _mean_center(new_log_diag)

        if use_adaptive_clip:
            new_log_diag = _adaptive_clip(new_log_diag, _max_cond)
        else:
            new_log_diag = _clip(new_log_diag, lo, hi)

        new_prev_grads = grads
        new_prev_params = params if params is not None else jax.tree.map(jnp.zeros_like, grads)

        new_state = SGDLearnableDiagCurvState(
            step=step,
            momentum=new_momentum,
            metric_ema=new_metric_ema,
            log_diag=new_log_diag,
            prev_grads=new_prev_grads,
            prev_params=new_prev_params,
        )
        return updates, new_state

    return optax.GradientTransformation(init, update)
