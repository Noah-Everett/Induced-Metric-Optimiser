# optimisers/jax_kronecker.py
"""
Bare-bones Kronecker-preconditioned optimizer for Dense-layer MLPs.

IMO-152: Prototype to validate the Kronecker direction before full theory.

For each Dense layer l with input x_l (B, in) and output gradient g_l (B, out):

    A_l = x_l.T @ x_l / B + damping * I    (input second moment)
    B_l = g_l.T @ g_l / B + damping * I    (output gradient second moment)
    update_W = A_l^{-1} @ grad_W @ B_l^{-1}  (preconditioned gradient)

No EMA, no spectral clipping, no embedding/denominator.

References:
- IMO-152: Kronecker prototype
- IMO-144: Kronecker rho factorization theorem
- Martens & Grosse (2015): Optimizing Neural Networks with KFAC
"""

from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import optax

from optimisers.jax_gn_diag import _activation_deriv


# ---- Kronecker factor computation -------------------------------------------

def kronecker_factors_mlp(params, x, y, activation_fn, output_grad_fn,
                          damping=1e-4,
                          layer_keys=('dense1', 'dense2', 'dense3')):
    """Compute damped Kronecker factors and their inverses for each layer.

    Parameters
    ----------
    params : dict
        Flax-style params: ``params['params'][layer_key]['kernel'|'bias']``.
    x : array (B, in_dim)
        Input batch.
    y : array (B,) or (B, out_dim)
        Labels.
    activation_fn : callable
        Element-wise activation for hidden layers.
    output_grad_fn : callable
        ``(logits, y) -> g_L`` of shape ``(B, out_dim)``.
    damping : float
        Tikhonov damping added to A and B before inversion.
    layer_keys : tuple of str
        Ordered layer names within ``params['params']``.

    Returns
    -------
    factors : dict
        ``{layer_key: (A_inv, B_inv)}`` where ``A_inv`` is ``(in_l, in_l)``
        and ``B_inv`` is ``(out_l, out_l)``.
    """
    p = params['params']
    B = x.shape[0]

    # Forward pass: collect layer inputs and pre-activations
    h = x
    layer_inputs = []
    pre_activations = []
    for i, key in enumerate(layer_keys):
        layer_inputs.append(h)
        z = h @ p[key]['kernel'] + p[key]['bias']
        pre_activations.append(z)
        if i < len(layer_keys) - 1:
            h = activation_fn(z)
        else:
            h = z

    logits = h
    g = output_grad_fn(logits, y)  # (B, out_dim)

    # Backward pass: compute Kronecker factors per layer
    factors = {}
    n_layers = len(layer_keys)
    for i in range(n_layers - 1, -1, -1):
        key = layer_keys[i]
        x_l = layer_inputs[i]  # (B, in_l)

        # Kronecker factors with damping
        A = x_l.T @ x_l / B + damping * jnp.eye(x_l.shape[1])
        B_mat = g.T @ g / B + damping * jnp.eye(g.shape[1])

        A_inv = jnp.linalg.inv(A)
        B_inv = jnp.linalg.inv(B_mat)

        factors[key] = (A_inv, B_inv)

        # Backprop g to previous layer
        if i > 0:
            g = g @ p[key]['kernel'].T
            g = g * _activation_deriv(activation_fn, pre_activations[i - 1])

    return factors


def apply_kronecker_preconditioner(grads, kron_factors, layer_keys):
    """Apply Kronecker preconditioning to gradients.

    For kernels:  update_W = A_inv @ grad_W @ B_inv
    For biases:   update_b = B_inv @ grad_b

    Parameters
    ----------
    grads : dict
        Gradient pytree matching params structure.
    kron_factors : dict
        ``{layer_key: (A_inv, B_inv)}`` from ``kronecker_factors_mlp``.
    layer_keys : tuple of str
        Layer names to precondition.

    Returns
    -------
    preconditioned : dict
        Same pytree structure as grads.
    """
    precond = {}
    for key in layer_keys:
        A_inv, B_inv = kron_factors[key]
        g_kernel = grads['params'][key]['kernel']
        g_bias = grads['params'][key]['bias']
        precond[key] = {
            'kernel': A_inv @ g_kernel @ B_inv,
            'bias': B_inv @ g_bias,
        }
    return {'params': precond}


# ---- Optimizer ---------------------------------------------------------------

class KroneckerNewtonState(NamedTuple):
    """State for the Kronecker Newton optimizer."""
    step: jnp.ndarray       # scalar int32
    momentum: Any            # pytree like params


def kronecker_newton(
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    layer_keys: tuple = ('dense1', 'dense2', 'dense3'),
) -> optax.GradientTransformation:
    """Bare-bones Kronecker-preconditioned optimizer.

    Momentum on raw gradients, then Kronecker preconditioning
    (matching the derived_newton_diag flow for fair comparison).

    The update function takes ``kron_factors`` as a positional argument::

        updates, state = opt.update(grads, state, kron_factors)

    Use :func:`kronecker_factors_mlp` to compute ``kron_factors``.

    Args:
        learning_rate: Parameter step size.
        momentum: Momentum coefficient (0 to disable).
        layer_keys: Ordered layer names in the param dict.
    """
    neg_lr = -learning_rate
    one_minus_mom = 1.0 - momentum
    use_momentum = momentum > 0.0

    def init(params):
        return KroneckerNewtonState(
            step=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
        )

    @jax.jit
    def update(grads, state, kron_factors, params=None):
        step = state.step + 1

        if use_momentum:
            # Momentum on raw gradients with bias correction
            new_mom = jax.tree.map(
                lambda m, g: momentum * m + one_minus_mom * g,
                state.momentum, grads,
            )
            m_corr = jax.tree.map(
                lambda m: m / (1.0 - momentum ** step),
                new_mom,
            )
        else:
            new_mom = state.momentum
            m_corr = grads

        # Apply Kronecker preconditioning
        updates = apply_kronecker_preconditioner(m_corr, kron_factors, layer_keys)

        # Scale by -lr
        updates = jax.tree.map(lambda u: neg_lr * u, updates)

        new_state = KroneckerNewtonState(step=step, momentum=new_mom)
        return updates, new_state

    return optax.GradientTransformation(init, update)


# ---- Convenience wrapper -----------------------------------------------------

def loss_grad_and_kronecker(params, x, y, loss_fn, activation_fn,
                            output_grad_fn, damping=1e-4,
                            layer_keys=('dense1', 'dense2', 'dense3')):
    """Compute loss, gradients, and Kronecker factors in one call.

    Parameters
    ----------
    params : dict
        Flax-style parameter dict.
    x : array (B, in_dim)
        Input batch.
    y : array
        Labels.
    loss_fn : callable
        ``loss_fn(params) -> scalar``.
    activation_fn : callable
        Element-wise activation for hidden layers.
    output_grad_fn : callable
        ``(logits, y) -> g_L``.
    damping : float
        Tikhonov damping for Kronecker factors.
    layer_keys : tuple of str
        Ordered layer names.

    Returns
    -------
    loss : scalar
    grads : dict (same structure as params)
    kron_factors : dict {layer_key: (A_inv, B_inv)}
    """
    loss, grads = jax.value_and_grad(loss_fn)(params)
    kron_factors = kronecker_factors_mlp(
        params, x, y, activation_fn, output_grad_fn,
        damping=damping, layer_keys=layer_keys,
    )
    return loss, grads, kron_factors
