# optimisers/jax_gn_diag.py
"""
Exact GN (Gauss-Newton) diagonal computation for Dense-layer MLPs.

Replaces the Hutchinson HVP estimator with a deterministic, per-layer
computation based on the KFAC diagonal formula:

    h_diag_W[i,j] = (1/B) sum_d  x_d[i]^2 * g_d[j]^2
    h_diag_b[j]   = (1/B) sum_d  g_d[j]^2

where x_d is the layer input and g_d = dL_d/dz_d is the per-sample
gradient of the loss w.r.t. the layer's pre-activation output.

This gives the diagonal of the empirical Fisher matrix, which equals
the GN diagonal for log-likelihood losses.  All values are >= 0 by
construction (squared quantities), so rho = 0 and neg_frac = 0.

Cost: ~1 forward pass + 1 backward pass + lightweight manual forward
and backward for the GN computation.  Total ~1.05x SGD, compared to
~3.8x SGD for Hutchinson on a 3-layer MLP.

The output pytree matches the Flax MLP params structure exactly, so
it plugs directly into the existing optimizer interface::

    updates, state = opt.update(grads, state, h_diag, params)

References:
- IMO-142: Replace Hutchinson with exact GN diagonal
- IMO-126: Hutchinson rho measurements (motivating this change)
"""

import jax
import jax.numpy as jnp


# ---- Output gradient functions -----------------------------------------------

def cross_entropy_output_grad(logits, y):
    """Per-sample gradient of cross-entropy loss w.r.t. logits.

    Parameters
    ----------
    logits : array (B, C)
        Pre-softmax logits.
    y : array (B,)
        Integer class labels.

    Returns
    -------
    g : array (B, C)
        Per-sample gradient: softmax(logits) - one_hot(y, C).
    """
    return jax.nn.softmax(logits, axis=-1) - jax.nn.one_hot(y, logits.shape[-1])


def mse_output_grad(logits, y):
    """Per-sample gradient of MSE loss w.r.t. pre-activation output.

    For loss_d = (pred_d - y_d)^2, the gradient w.r.t. the
    pre-activation (identity output activation) is 2*(pred_d - y_d).

    Parameters
    ----------
    logits : array (B, out_dim) or (B,)
        Model output (pre-activation).
    y : array (B,) or (B, out_dim)
        Regression targets.

    Returns
    -------
    g : array (B, out_dim)
        Per-sample gradient.
    """
    if logits.ndim == 1:
        return 2.0 * (logits - y.reshape(-1))[:, None]
    if logits.ndim == 2 and logits.shape[-1] == 1:
        # output_dim=1: flatten both to (B,) to avoid broadcast mismatch
        y_flat = y.reshape(-1)
        return 2.0 * (logits.squeeze(-1) - y_flat)[:, None]
    return 2.0 * (logits - y)


# ---- Activation derivative ---------------------------------------------------

def _activation_deriv(activation_fn, z):
    """Element-wise derivative of an activation function at z.

    Uses forward-mode AD: jvp with a ones tangent gives the
    element-wise derivative for any element-wise function.

    Parameters
    ----------
    activation_fn : callable
        Element-wise activation (e.g., jax.nn.gelu).
    z : array
        Pre-activation values.

    Returns
    -------
    deriv : array (same shape as z)
        Element-wise derivative activation_fn'(z).
    """
    _, deriv = jax.jvp(activation_fn, (z,), (jnp.ones_like(z),))
    return deriv


# ---- Core GN diagonal --------------------------------------------------------

def gn_diag_mlp(params, x, y, activation_fn, output_grad_fn,
                layer_keys=('dense1', 'dense2', 'dense3')):
    """Compute exact GN diagonal for a Dense-layer MLP.

    Performs a manual forward pass to collect per-layer activations,
    computes the output gradient, then backpropagates squared
    quantities to obtain the GN diagonal for each layer's kernel
    and bias.

    Parameters
    ----------
    params : dict
        Flax-style params: ``params['params'][layer_key]['kernel'|'bias']``.
    x : array (B, in_dim)
        Input batch.
    y : array (B,) or (B, out_dim)
        Labels (integer for cross-entropy, float for MSE).
    activation_fn : callable
        Element-wise activation applied after all hidden layers
        (not the output layer).
    output_grad_fn : callable
        ``(logits, y) -> g_L`` of shape ``(B, out_dim)``.
        Per-sample gradient of the loss w.r.t. output pre-activations.
    layer_keys : tuple of str
        Ordered layer names within ``params['params']``.

    Returns
    -------
    h_diag : dict
        Same pytree structure as ``params``.
    """
    p = params['params']
    B = x.shape[0]

    for key in layer_keys:
        if key not in p:
            raise ValueError(
                f"layer_keys contains '{key}' but params['params'] "
                f"only has: {sorted(p.keys())}")

    # --- Forward pass: collect layer inputs and pre-activations ---
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
            h = z  # no activation on output layer

    logits = h

    # --- Output gradient ---
    g = output_grad_fn(logits, y)  # (B, out_dim)

    # --- Backward pass: compute GN diagonal per layer ---
    h_diag_params = {}
    n_layers = len(layer_keys)
    for i in range(n_layers - 1, -1, -1):
        key = layer_keys[i]
        x_l = layer_inputs[i]  # (B, in_dim_l)

        # GN diagonal for this layer
        h_diag_kernel = (x_l ** 2).T @ (g ** 2) / B  # (in_l, out_l)
        h_diag_bias = jnp.mean(g ** 2, axis=0)        # (out_l,)
        h_diag_params[key] = {'kernel': h_diag_kernel, 'bias': h_diag_bias}

        # Backprop g to previous layer (if not the first layer)
        if i > 0:
            g = g @ p[key]['kernel'].T                    # through linear
            g = g * _activation_deriv(activation_fn, pre_activations[i - 1])  # through activation

    return {'params': h_diag_params}


def loss_grad_and_gn_diag(params, x, y, loss_fn, activation_fn,
                          output_grad_fn,
                          layer_keys=('dense1', 'dense2', 'dense3')):
    """Compute loss, parameter gradients, and exact GN diagonal.

    Uses ``jax.value_and_grad`` for loss + grads (one forward + backward
    through the Flax model), then ``gn_diag_mlp`` for the GN diagonal
    (lightweight manual forward + backward, no full AD graph).

    Parameters
    ----------
    params : dict
        Flax-style parameter dict.
    x : array (B, in_dim)
        Input batch.
    y : array
        Labels.
    loss_fn : callable
        ``loss_fn(params) -> scalar``.  Typically a closure over
        ``x, y, model``: ``lambda p: loss_fn(p, x, y, model)``.
    activation_fn : callable
        Element-wise activation for hidden layers.
    output_grad_fn : callable
        ``(logits, y) -> g_L`` of shape ``(B, out_dim)``.
    layer_keys : tuple of str
        Ordered layer names.

    Returns
    -------
    loss : scalar
    grads : dict (same structure as params)
    h_diag : dict (same structure as params)
    """
    loss, grads = jax.value_and_grad(loss_fn)(params)
    h_diag = gn_diag_mlp(params, x, y, activation_fn, output_grad_fn,
                         layer_keys)
    return loss, grads, h_diag
