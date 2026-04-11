"""
Diagnostic logging for optimizer internal state over training.

Provides ``collect_diagnostics()`` — inspects optimizer state by name and
returns a flat dict of metrics (prefixed ``diag/``) suitable for
``logger.log()`` or ``wandb.log()``.

Enabled via ``--diagnostics`` CLI flag.  Off by default.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Pytree helpers
# ---------------------------------------------------------------------------

def _scalar(x):
    """JAX/numpy scalar -> Python float."""
    if hasattr(x, 'item'):
        return float(x.item())
    return float(x)


def _path_str(key_path):
    """JAX key path -> ``params/Dense_0/kernel`` style string."""
    parts = []
    for k in key_path:
        if hasattr(k, 'key'):
            parts.append(str(k.key))
        elif hasattr(k, 'idx'):
            parts.append(str(k.idx))
        else:
            parts.append(str(k))
    return '/'.join(parts) or 'root'


def _tree_norm(tree):
    """Global L2 norm of a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    return _scalar(jnp.sqrt(sum(jnp.sum(l ** 2) for l in leaves)))


def _tree_dot(tree_a, tree_b):
    """Dot product across two pytrees of matching structure."""
    a_leaves = jax.tree_util.tree_leaves(tree_a)
    b_leaves = jax.tree_util.tree_leaves(tree_b)
    return sum(_scalar(jnp.sum(a * b)) for a, b in zip(a_leaves, b_leaves))


def _count_params(tree):
    """Total number of scalar elements in a pytree."""
    return sum(l.size for l in jax.tree_util.tree_leaves(tree))


def _global_stats(tree, prefix):
    """Mean / std / min / max / norm across all leaves."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return {}
    all_flat = jnp.concatenate([l.ravel() for l in leaves])
    return {
        f'{prefix}/mean': _scalar(jnp.mean(all_flat)),
        f'{prefix}/std':  _scalar(jnp.std(all_flat)),
        f'{prefix}/min':  _scalar(jnp.min(all_flat)),
        f'{prefix}/max':  _scalar(jnp.max(all_flat)),
        f'{prefix}/norm': _scalar(jnp.sqrt(jnp.sum(all_flat ** 2))),
    }


def _per_leaf_stats(tree, prefix):
    """Mean / std / min / max per leaf (per layer)."""
    metrics = {}
    try:
        key_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    except Exception:
        return metrics
    for key_path, leaf in key_leaves:
        p = f'{prefix}/{_path_str(key_path)}'
        flat = leaf.ravel()
        metrics[f'{p}/mean'] = _scalar(jnp.mean(flat))
        metrics[f'{p}/std']  = _scalar(jnp.std(flat))
        metrics[f'{p}/min']  = _scalar(jnp.min(flat))
        metrics[f'{p}/max']  = _scalar(jnp.max(flat))
    return metrics


def _full_tensor(tree, prefix):
    """Log individual values for small pytrees (2D problems, etc.)."""
    metrics = {}
    try:
        key_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    except Exception:
        leaves = jax.tree_util.tree_leaves(tree)
        if leaves and leaves[0].size <= 100:
            for i, v in enumerate(leaves[0].ravel().tolist()):
                metrics[f'{prefix}/{i}'] = v
        return metrics
    for key_path, leaf in key_leaves:
        p = f'{prefix}/{_path_str(key_path)}'
        if leaf.size <= 100:
            for i, v in enumerate(leaf.ravel().tolist()):
                metrics[f'{p}/{i}'] = v
    return metrics


# ---------------------------------------------------------------------------
# Common metric extraction (shared by all custom optimizers)
# ---------------------------------------------------------------------------

def _metric_common(step_val, metric_ema, momentum_tree, config, is_log=False):
    """metric_ema -> v_hat -> r, plus momentum stats."""
    m = {}
    beta = config.get('beta', 0.8)

    m['metric_ema'] = _scalar(metric_ema)

    if step_val > 0 and beta < 1.0:
        v_hat = _scalar(metric_ema) / (1.0 - beta ** step_val)
    else:
        v_hat = _scalar(metric_ema)
    m['v_hat'] = v_hat

    if not is_log:
        m['r'] = 1.0 / (1.0 + abs(v_hat))

    m.update(_global_stats(momentum_tree, 'mom'))
    m.update(_per_leaf_stats(momentum_tree, 'mom'))
    return m


# ---------------------------------------------------------------------------
# Per-family extractors
# ---------------------------------------------------------------------------

def _extract_optax_sgd(opt_state, config):
    m = {}
    try:
        inner = opt_state[0] if isinstance(opt_state, tuple) else opt_state
        if hasattr(inner, 'trace'):
            m.update(_global_stats(inner.trace, 'mom'))
            m.update(_per_leaf_stats(inner.trace, 'mom'))
    except Exception:
        pass
    return m


def _extract_optax_adam(opt_state, config):
    m = {}
    try:
        adam = opt_state[0] if isinstance(opt_state, tuple) else opt_state

        if hasattr(adam, 'count'):
            count = _scalar(adam.count)
            m['step'] = count
        else:
            count = 1

        if hasattr(adam, 'mu'):
            m.update(_global_stats(adam.mu, 'mu'))
            m.update(_per_leaf_stats(adam.mu, 'mu'))

        if hasattr(adam, 'nu'):
            m.update(_global_stats(adam.nu, 'nu'))
            m.update(_per_leaf_stats(adam.nu, 'nu'))

            # Bias-corrected second moment nu_hat
            b2 = config.get('beta2', config.get('b2', 0.999))
            eps = config.get('eps', 1e-8)
            lr = config.get('learning_rate', 0.001)
            if count > 0:
                bc2 = 1.0 - b2 ** count
                nu_leaves = jax.tree_util.tree_leaves(adam.nu)
                nu_hat_all = jnp.concatenate([
                    (nu_l / bc2).ravel() for nu_l in nu_leaves
                ])
                m['nu_hat/mean'] = _scalar(jnp.mean(nu_hat_all))
                m['nu_hat/std']  = _scalar(jnp.std(nu_hat_all))
                m['nu_hat/min']  = _scalar(jnp.min(nu_hat_all))
                m['nu_hat/max']  = _scalar(jnp.max(nu_hat_all))

                # Effective LR:  eta / (sqrt(nu_hat) + eps)
                eff_lr_all = lr / (jnp.sqrt(nu_hat_all) + eps)
                m['eff_lr/mean'] = _scalar(jnp.mean(eff_lr_all))
                m['eff_lr/std']  = _scalar(jnp.std(eff_lr_all))
                m['eff_lr/min']  = _scalar(jnp.min(eff_lr_all))
                m['eff_lr/max']  = _scalar(jnp.max(eff_lr_all))
    except Exception:
        pass
    return m


def _extract_fixed(opt_state, config, is_log=False, is_rms=False):
    """SGDState — sgd_metric, sgd_log_metric, sgd_rms."""
    step_val = _scalar(opt_state.step)
    m = _metric_common(step_val, opt_state.metric_ema,
                        opt_state.momentum, config, is_log)

    if is_rms and opt_state.rms_ema is not None:
        m.update(_global_stats(opt_state.rms_ema, 'rms_ema'))
        m.update(_per_leaf_stats(opt_state.rms_ema, 'rms_ema'))

        # Bias-corrected rms_corrected
        beta_rms = config.get('beta_rms', 0.99)
        eps = config.get('eps', 1e-8)
        if step_val > 0 and beta_rms < 1.0:
            rms_bc = 1.0 - beta_rms ** step_val
            rms_leaves = jax.tree_util.tree_leaves(opt_state.rms_ema)
            rms_corr_all = jnp.concatenate([
                (r_l / rms_bc).ravel() for r_l in rms_leaves
            ])
            m['rms_corrected/mean'] = _scalar(jnp.mean(rms_corr_all))
            m['rms_corrected/std']  = _scalar(jnp.std(rms_corr_all))
            m['rms_corrected/min']  = _scalar(jnp.min(rms_corr_all))
            m['rms_corrected/max']  = _scalar(jnp.max(rms_corr_all))

            # Per-layer condition number: max(sqrt(rms)) / min(sqrt(rms))
            try:
                kl, _ = jax.tree_util.tree_flatten_with_path(opt_state.rms_ema)
                for kp, leaf in kl:
                    corr = jnp.sqrt(leaf / rms_bc + eps)
                    c_max = _scalar(jnp.max(corr))
                    c_min = max(_scalar(jnp.min(corr)), 1e-30)
                    m[f'rms_condition/{_path_str(kp)}'] = c_max / c_min
            except Exception:
                pass
    return m


def _extract_learnable_scalar(opt_state, config, is_log=False, grads=None):
    """SGDLearnableScalarState."""
    step_val = _scalar(opt_state.step)
    m = _metric_common(step_val, opt_state.metric_ema,
                        opt_state.momentum, config, is_log)

    s = _scalar(opt_state.log_scale)
    alpha = float(np.exp(np.clip(s, -20, 20)))
    m['log_scale'] = s
    m['alpha'] = alpha

    lr = config.get('learning_rate', 0.1)
    if 'r' in m:
        m['eff_lr'] = lr * m['r'] * alpha

    # grad_s = xi * exp(s) * sum(g^2)
    if grads is not None:
        xi = config.get('xi', 0.1)
        g_sq_sum = sum(_scalar(jnp.sum(g ** 2))
                       for g in jax.tree_util.tree_leaves(grads))
        m['grad_s'] = xi * alpha * g_sq_sum

    return m


def _extract_learnable_diag(opt_state, config, is_log=False, grads=None):
    """SGDLearnableDiagState (also serves as base for curv variant)."""
    step_val = _scalar(opt_state.step)
    m = _metric_common(step_val, opt_state.metric_ema,
                        opt_state.momentum, config, is_log)

    # --- log_diag stats ---
    m.update(_global_stats(opt_state.log_diag, 'log_diag'))
    m.update(_per_leaf_stats(opt_state.log_diag, 'log_diag'))

    lr = config.get('learning_rate', 0.1)
    xi = config.get('xi', 0.1)
    metric_clip = config.get('metric_clip', 4.0)
    metric_lr = config.get('metric_lr', 1e-3)
    metric_reg = config.get('metric_reg', 1e-4)

    diag_leaves = jax.tree_util.tree_leaves(opt_state.log_diag)
    all_s = jnp.concatenate([l.ravel() for l in diag_leaves])
    all_exp_s = jnp.exp(all_s)

    # Effective LR per parameter: lr * r * exp(s_i)
    if 'r' in m:
        eff_lr = lr * m['r'] * all_exp_s
        m['eff_lr/mean'] = _scalar(jnp.mean(eff_lr))
        m['eff_lr/std']  = _scalar(jnp.std(eff_lr))
        m['eff_lr/min']  = _scalar(jnp.min(eff_lr))
        m['eff_lr/max']  = _scalar(jnp.max(eff_lr))

    # Metric condition number
    m['metric_condition'] = (_scalar(jnp.max(all_exp_s))
                             / max(_scalar(jnp.min(all_exp_s)), 1e-30))

    # Per-layer condition numbers
    try:
        kl, _ = jax.tree_util.tree_flatten_with_path(opt_state.log_diag)
        for kp, leaf in kl:
            es = jnp.exp(leaf.ravel())
            m[f'metric_condition/{_path_str(kp)}'] = (
                _scalar(jnp.max(es)) / max(_scalar(jnp.min(es)), 1e-30)
            )
    except Exception:
        pass

    # Clipped fraction
    clip_bound = abs(metric_clip) - 1e-6
    m['clipped_frac'] = _scalar(jnp.mean(jnp.abs(all_s) >= clip_bound))

    # Metric entropy
    n = all_s.size
    if n > 1:
        p = all_exp_s / jnp.sum(all_exp_s)
        entropy = _scalar(-jnp.sum(p * jnp.log(p + 1e-30)))
        m['metric_entropy'] = entropy
        m['metric_entropy_norm'] = entropy / float(np.log(n))

    # --- Quantities requiring grads ---
    if grads is not None:
        g_leaves = jax.tree_util.tree_leaves(grads)
        s_leaves = diag_leaves

        # g_tilde = exp(s) * g  (preconditioned gradient)
        g_tilde_parts = [jnp.exp(s) * g for s, g in zip(s_leaves, g_leaves)]
        m.update(_global_stats(g_tilde_parts, 'g_tilde'))

        # grad_s = xi * exp(s) * g^2  (metric learning gradient)
        grad_s_parts = [xi * jnp.exp(s) * (g ** 2)
                        for s, g in zip(s_leaves, g_leaves)]
        all_grad_s = jnp.concatenate([gs.ravel() for gs in grad_s_parts])
        m['grad_s/mean'] = _scalar(jnp.mean(all_grad_s))
        m['grad_s/std']  = _scalar(jnp.std(all_grad_s))
        m['grad_s/min']  = _scalar(jnp.min(all_grad_s))
        m['grad_s/max']  = _scalar(jnp.max(all_grad_s))

        # Mean-centering offset:
        # After s_new = s + mu*grad_s - mu*reg*s, centering removes mean(s_new).
        # offset ≈ mu * mean(grad_s) - mu * reg * mean(s)
        m['mean_center_offset'] = (
            metric_lr * _scalar(jnp.mean(all_grad_s))
            - metric_lr * metric_reg * _scalar(jnp.mean(all_s))
        )

    return m


def _extract_learnable_diag_curv(opt_state, config, is_log=False,
                                  grads=None, params=None):
    """SGDLearnableDiagCurvState — extends diagonal with curvature."""
    m = _extract_learnable_diag(opt_state, config, is_log, grads)

    step_val = _scalar(opt_state.step)
    if step_val <= 1 or grads is None or params is None:
        return m

    secant_eps = config.get('secant_eps', 1e-5)
    curv_tau = config.get('curv_tau', 1.0)
    curv_beta = config.get('curv_beta', 0.05)
    xi = config.get('xi', 0.1)

    g_leaves  = jax.tree_util.tree_leaves(grads)
    pg_leaves = jax.tree_util.tree_leaves(opt_state.prev_grads)
    p_leaves  = jax.tree_util.tree_leaves(params)
    pp_leaves = jax.tree_util.tree_leaves(opt_state.prev_params)
    s_leaves  = jax.tree_util.tree_leaves(opt_state.log_diag)

    all_h, all_cs, all_cc, all_bsg = [], [], [], []

    for g, pg, p, pp, s in zip(g_leaves, pg_leaves, p_leaves, pp_leaves, s_leaves):
        dt = p - pp
        dg = g - pg
        safe = jnp.abs(dt) > secant_eps
        denom = jnp.where(safe, dt, jnp.ones_like(dt))
        h_hat = jnp.where(safe, dg / denom, jnp.zeros_like(g))
        all_h.append(h_hat.ravel())

        curv_sign = jnp.tanh(h_hat / curv_tau)
        all_cs.append(curv_sign.ravel())

        cc = curv_beta * curv_sign * jnp.exp(s) * (g ** 2)
        all_cc.append(cc.ravel())

        bsg = xi * jnp.exp(s) * (g ** 2)
        all_bsg.append(bsg.ravel())

    all_h   = jnp.concatenate(all_h)
    all_cs  = jnp.concatenate(all_cs)
    all_cc  = jnp.concatenate(all_cc)
    all_bsg = jnp.concatenate(all_bsg)

    # Secant Hessian diagonal
    m['H_hat/mean'] = _scalar(jnp.mean(all_h))
    m['H_hat/std']  = _scalar(jnp.std(all_h))
    m['H_hat/min']  = _scalar(jnp.min(all_h))
    m['H_hat/max']  = _scalar(jnp.max(all_h))
    m['positive_curv_frac'] = _scalar(jnp.mean(all_h > 0))

    # Curvature sign: tanh(H/tau)
    m['curv_sign/mean'] = _scalar(jnp.mean(all_cs))
    m['curv_sign/std']  = _scalar(jnp.std(all_cs))

    # Curvature correction magnitude
    cc_abs = jnp.abs(all_cc)
    m['curv_corr/mean'] = _scalar(jnp.mean(cc_abs))
    m['curv_corr/max']  = _scalar(jnp.max(cc_abs))

    # Correction ratio: |curv_corr| / |base_s_grad|
    bsg_mean = max(_scalar(jnp.mean(jnp.abs(all_bsg))), 1e-30)
    m['curv_corr_ratio'] = _scalar(jnp.mean(cc_abs)) / bsg_mean

    return m


_OFFDIAG_MODE = {'0': 'zero', 'l': 'grad', 'm': 'momentum', 'theta': 'params'}


def _extract_offdiag(opt_state, config, optimizer_name,
                      grads=None, params=None):
    """OffDiagSGDState — momentum + Woodbury 2x2 diagnostics."""
    m = {}
    step_val = _scalar(opt_state.step)

    m.update(_global_stats(opt_state.momentum, 'mom'))
    m.update(_per_leaf_stats(opt_state.momentum, 'mom'))

    if step_val <= 0 or grads is None or params is None:
        return m

    xi_val    = _scalar(opt_state.xi)
    gamma_val = _scalar(opt_state.gamma)
    mom_coeff = _scalar(opt_state.mom_coeff)

    bc = 1.0 - mom_coeff ** step_val
    m_hat = jax.tree.map(lambda v: v / bc, opt_state.momentum)

    # Parse modes from name: sgd_offdiag_{a}_{b}
    parts = optimizer_name.split('_')
    a_mode = _OFFDIAG_MODE.get(parts[2], 'zero')
    b_mode = _OFFDIAG_MODE.get(parts[3], 'zero')
    base_mode = config.get('base_mode', 'grad')

    l_tree = grads if base_mode in ('grad', 'g', 'gradient', 'l') else m_hat

    def _vec(mode):
        if mode == 'zero':
            return opt_state.zero_buffer
        if mode == 'grad':
            return grads
        if mode == 'momentum':
            return m_hat
        if mode == 'params':
            return params
        return opt_state.zero_buffer

    a_tree = _vec(a_mode)
    b_tree = _vec(b_mode)

    inv_gamma = 1.0 / gamma_val
    alpha = xi_val * inv_gamma
    bl_tree = jax.tree.map(lambda b, l: b + l, b_tree, l_tree)

    M00 = 1.0 + alpha * _tree_dot(l_tree, a_tree)
    M01 = alpha * _tree_dot(l_tree, l_tree)
    M10 = alpha * _tree_dot(bl_tree, a_tree)
    M11 = 1.0 + alpha * _tree_dot(bl_tree, l_tree)
    det = M00 * M11 - M01 * M10

    m['M00'] = M00
    m['M01'] = M01
    m['M10'] = M10
    m['M11'] = M11
    m['woodbury_det'] = det

    if abs(det) > 1e-8:
        use_mom = config.get('use_momentum_for_update', True)
        r_tree = m_hat if use_mom else l_tree
        z_tree = jax.tree.map(lambda r: inv_gamma * r, r_tree)
        rhs0 = _tree_dot(l_tree, z_tree)
        rhs1 = _tree_dot(bl_tree, z_tree)
        inv_det = 1.0 / det
        m['cramer_w0'] = inv_det * (M11 * rhs0 - M01 * rhs1)
        m['cramer_w1'] = inv_det * (M00 * rhs1 - M10 * rhs0)

    return m


# ---------------------------------------------------------------------------
# Momentum-gradient alignment
# ---------------------------------------------------------------------------

def _mom_grad_alignment(momentum_tree, grads):
    """cos(momentum, gradient)."""
    if grads is None or momentum_tree is None:
        return {}
    m_norm = _tree_norm(momentum_tree)
    g_norm = _tree_norm(grads)
    if m_norm < 1e-30 or g_norm < 1e-30:
        return {'mom_grad_cos': 0.0}
    dot = _tree_dot(momentum_tree, grads)
    return {'mom_grad_cos': dot / (m_norm * g_norm)}


# ---------------------------------------------------------------------------
# Training-level metrics
# ---------------------------------------------------------------------------

def _training_metrics(grads, params, updates, loss=None, prev_loss=None):
    m = {}

    if grads is not None:
        m['grad_norm'] = _tree_norm(grads)
        m.update(_per_leaf_stats(grads, 'grad'))

    if params is not None:
        m['param_norm'] = _tree_norm(params)

    if updates is not None:
        un = _tree_norm(updates)
        m['update_norm'] = un
        if params is not None:
            pn = _tree_norm(params)
            m['relative_step'] = un / max(pn, 1e-30)
        if grads is not None:
            gn = _tree_norm(grads)
            m['eff_lr_empirical'] = un / max(gn, 1e-30)

    if loss is not None:
        m['loss_at_diag'] = _scalar(loss) if hasattr(loss, 'item') else float(loss)

    if loss is not None and prev_loss is not None:
        pl = _scalar(prev_loss) if hasattr(prev_loss, 'item') else float(prev_loss)
        cl = _scalar(loss) if hasattr(loss, 'item') else float(loss)
        if abs(pl) > 1e-30:
            m['loss_change'] = (cl - pl) / abs(pl)

    return m


# ---------------------------------------------------------------------------
# Family dispatch
# ---------------------------------------------------------------------------

def _get_family(name):
    if name == 'sgd':
        return 'optax_sgd'
    if name in ('adam', 'adamw'):
        return 'optax_adam'
    if name == 'muon':
        return 'optax_muon'
    if name == 'sgd_metric':
        return 'fixed'
    if name == 'sgd_log_metric':
        return 'fixed_log'
    if name == 'sgd_rms':
        return 'rms'
    if name == 'sgd_learn_scalar':
        return 'scalar'
    if name == 'sgd_learn_scalar_log':
        return 'scalar_log'
    if name == 'sgd_learn_diag':
        return 'diag'
    if name == 'sgd_learn_diag_log':
        return 'diag_log'
    if name == 'sgd_learn_diag_curv':
        return 'curv'
    if name == 'sgd_learn_diag_curv_log':
        return 'curv_log'
    if name.startswith('sgd_offdiag_'):
        return 'offdiag'
    return 'unknown'


def _get_momentum_tree(family, opt_state):
    """Return the momentum pytree from optimizer state, or None."""
    if family == 'optax_sgd':
        try:
            inner = opt_state[0] if isinstance(opt_state, tuple) else opt_state
            return inner.trace if hasattr(inner, 'trace') else None
        except Exception:
            return None
    if family in ('optax_adam', 'optax_muon'):
        try:
            inner = opt_state[0] if isinstance(opt_state, tuple) else opt_state
            return inner.mu if hasattr(inner, 'mu') else None
        except Exception:
            return None
    if hasattr(opt_state, 'momentum'):
        return opt_state.momentum
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_diagnostics(
    optimizer_name: str,
    opt_state: Any,
    params: Any,
    grads: Any = None,
    updates: Any = None,
    loss: Any = None,
    config: Optional[Dict] = None,
    prev_loss: Any = None,
) -> Dict[str, float]:
    """Collect diagnostic metrics from optimizer state.

    Parameters
    ----------
    optimizer_name : str
        Name from ``ALL_OPTIMIZERS`` registry.
    opt_state
        Current optimizer state.
    params
        Current model parameters.
    grads
        Gradients at current point (enables training-level metrics).
    updates
        Parameter updates from optimizer.
    loss
        Current loss value.
    config : dict, optional
        Hyperparameter config (needed for bias correction, etc.).
    prev_loss
        Previous diagnostic loss (for relative change).

    Returns
    -------
    dict
        Flat dict of ``diag/key -> float``.
    """
    config = config or {}
    metrics = {}
    family = _get_family(optimizer_name)

    # --- Optimizer state ---
    if family == 'optax_sgd':
        metrics.update(_extract_optax_sgd(opt_state, config))
    elif family in ('optax_adam', 'optax_muon'):
        metrics.update(_extract_optax_adam(opt_state, config))
    elif family in ('fixed', 'fixed_log'):
        metrics.update(_extract_fixed(opt_state, config,
                                       is_log=('log' in family)))
    elif family == 'rms':
        metrics.update(_extract_fixed(opt_state, config, is_rms=True))
    elif family in ('scalar', 'scalar_log'):
        metrics.update(_extract_learnable_scalar(
            opt_state, config, is_log=('log' in family), grads=grads))
    elif family in ('diag', 'diag_log'):
        metrics.update(_extract_learnable_diag(
            opt_state, config, is_log=('log' in family), grads=grads))
    elif family in ('curv', 'curv_log'):
        metrics.update(_extract_learnable_diag_curv(
            opt_state, config, is_log=('log' in family),
            grads=grads, params=params))
    elif family == 'offdiag':
        metrics.update(_extract_offdiag(
            opt_state, config, optimizer_name,
            grads=grads, params=params))

    # --- Momentum-gradient alignment ---
    mom = _get_momentum_tree(family, opt_state)
    if mom is not None:
        metrics.update(_mom_grad_alignment(mom, grads))

    # --- Training-level ---
    metrics.update(_training_metrics(grads, params, updates,
                                      loss, prev_loss))

    # --- Full tensor logging for small problems ---
    n = _count_params(params)
    if n <= 1000:
        if hasattr(opt_state, 'log_diag'):
            metrics.update(_full_tensor(opt_state.log_diag, 'full/log_diag'))
        if hasattr(opt_state, 'log_scale'):
            metrics['full/log_scale'] = _scalar(opt_state.log_scale)
        if hasattr(opt_state, 'momentum'):
            metrics.update(_full_tensor(opt_state.momentum, 'full/mom'))
        if grads is not None:
            metrics.update(_full_tensor(grads, 'full/grad'))
        if updates is not None:
            metrics.update(_full_tensor(updates, 'full/update'))
        if params is not None:
            metrics.update(_full_tensor(params, 'full/param'))

        # Curv-specific full tensors: per-component H_hat and curv_sign
        if (family in ('curv', 'curv_log')
                and hasattr(opt_state, 'prev_grads')
                and hasattr(opt_state, 'prev_params')
                and grads is not None and params is not None):
            step_val = _scalar(opt_state.step)
            if step_val > 1:
                secant_eps = config.get('secant_eps', 1e-5)
                curv_tau = config.get('curv_tau', 1.0)
                g_leaves = jax.tree_util.tree_leaves(grads)
                pg_leaves = jax.tree_util.tree_leaves(opt_state.prev_grads)
                p_leaves = jax.tree_util.tree_leaves(params)
                pp_leaves = jax.tree_util.tree_leaves(opt_state.prev_params)
                h_parts, cs_parts = [], []
                for g, pg, p, pp in zip(g_leaves, pg_leaves, p_leaves, pp_leaves):
                    dt = p - pp
                    dg = g - pg
                    safe = jnp.abs(dt) > secant_eps
                    denom = jnp.where(safe, dt, jnp.ones_like(dt))
                    h = jnp.where(safe, dg / denom, jnp.zeros_like(g))
                    h_parts.append(h.ravel())
                    cs_parts.append(jnp.tanh(h / curv_tau).ravel())
                all_h = jnp.concatenate(h_parts)
                all_cs = jnp.concatenate(cs_parts)
                for i, v in enumerate(all_h.tolist()):
                    metrics[f'full/H_hat/{i}'] = v
                for i, v in enumerate(all_cs.tolist()):
                    metrics[f'full/curv_sign/{i}'] = v

    return {f'diag/{k}': v for k, v in metrics.items()}


def diagnostic_step(
    optimizer_name: str,
    optimizer,
    opt_state: Any,
    params: Any,
    grads: Any,
    loss: Any = None,
    config: Optional[Dict] = None,
    use_loss: bool = False,
    prev_loss: Any = None,
) -> Dict[str, float]:
    """Compute updates from grads/state and collect all diagnostics.

    Convenience wrapper: calls ``optimizer.update()`` to get updates
    (without stepping), then ``collect_diagnostics()``.
    """
    if use_loss and loss is not None:
        updates, _ = optimizer.update(grads, opt_state, loss, params)
    else:
        updates, _ = optimizer.update(grads, opt_state, params)

    return collect_diagnostics(
        optimizer_name, opt_state, params,
        grads=grads, updates=updates, loss=loss,
        config=config, prev_loss=prev_loss,
    )
