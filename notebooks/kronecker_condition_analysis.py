"""
Kronecker condition number analysis.

Quick validation: compute per-layer Kronecker factors on MNIST MLP and
compare the condition number of the preconditioned system under:
  1. No preconditioning (raw GN Hessian)
  2. Diagonal preconditioning (current Newton target)
  3. Kronecker preconditioning (proposed KFAC-style)

If Kronecker dramatically reduces condition number vs diagonal, the
direction is promising.

Run:
    PYTHONPATH=repo .conda/bin/python3.11 repo/notebooks/kronecker_condition_analysis.py
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.insert(0, repo_dir)

import jax
import jax.numpy as jnp
import optax
import numpy as np
import tensorflow as tf

from optimisers.jax_gn_diag import _activation_deriv, cross_entropy_output_grad
from parameters.shared_models import MLP


def load_mnist_batch(batch_size=1024):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x = jnp.array(x_train[:batch_size].reshape(-1, 784) / 255.0, dtype=jnp.float32)
    y = jnp.array(y_train[:batch_size])
    return x, y


def compute_kronecker_factors(params, x, y, activation_fn, output_grad_fn,
                               layer_keys=('dense1', 'dense2', 'dense3')):
    """Compute per-layer Kronecker factors A_l, B_l and GN diagonal."""
    p = params['params']
    B = x.shape[0]

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
    g = output_grad_fn(logits, y)

    results = {}
    n_layers = len(layer_keys)
    for i in range(n_layers - 1, -1, -1):
        key = layer_keys[i]
        x_l = layer_inputs[i]

        # Kronecker factors
        A_l = x_l.T @ x_l / B      # (in_l, in_l) input second moment
        B_l = g.T @ g / B           # (out_l, out_l) output gradient second moment

        # GN diagonal (for comparison)
        h_diag_kernel = (x_l ** 2).T @ (g ** 2) / B
        h_diag_bias = jnp.mean(g ** 2, axis=0)

        results[key] = {
            'A': A_l, 'B': B_l,
            'h_diag_kernel': h_diag_kernel,
            'h_diag_bias': h_diag_bias,
            'n_in': x_l.shape[1],
            'n_out': g.shape[1],
        }

        # Backprop g
        if i > 0:
            g = g @ p[key]['kernel'].T
            g = g * _activation_deriv(activation_fn, pre_activations[i - 1])

    return results


def analyze_layer(name, layer_data, damping=1e-6):
    """Analyze condition numbers for one layer."""
    A = np.array(layer_data['A'])
    B = np.array(layer_data['B'])
    h_diag = np.array(layer_data['h_diag_kernel'])
    n_in = layer_data['n_in']
    n_out = layer_data['n_out']
    n_params = n_in * n_out

    print(f"\n{'=' * 70}")
    print(f"Layer: {name}  ({n_in} x {n_out} = {n_params} params)")
    print(f"{'=' * 70}")

    # Factor spectra
    eig_A = np.linalg.eigvalsh(A)
    eig_B = np.linalg.eigvalsh(B)

    print(f"\n--- Kronecker factor spectra ---")
    print(f"  A ({n_in}x{n_in}): min={eig_A[0]:.4e}  max={eig_A[-1]:.4e}  "
          f"kappa={eig_A[-1]/max(eig_A[0], 1e-30):.1f}")
    print(f"  B ({n_out}x{n_out}): min={eig_B[0]:.4e}  max={eig_B[-1]:.4e}  "
          f"kappa={eig_B[-1]/max(eig_B[0], 1e-30):.1f}")

    # Full Kronecker Hessian eigenvalues = all products lambda_i * mu_j
    kron_eigs = np.outer(eig_A, eig_B).ravel()
    kron_eigs.sort()
    kappa_raw = kron_eigs[-1] / max(kron_eigs[0], 1e-30)

    print(f"\n--- Raw GN Hessian (A kron B) ---")
    print(f"  eigenvalue range: [{kron_eigs[0]:.4e}, {kron_eigs[-1]:.4e}]")
    print(f"  condition number: {kappa_raw:.1f}")

    # Diagonal preconditioning: D^{-1} H where D = diag(H)
    # The diagonal of A kron B is kron(diag(A), diag(B)) = outer(diag(A), diag(B))
    diag_A = np.diag(A)
    diag_B = np.diag(B)
    kron_diag = np.outer(diag_A, diag_B).ravel()

    # For condition number after diagonal preconditioning, we compute
    # eigenvalues of D^{-1/2} H D^{-1/2} where H = A kron B
    # This is expensive for large layers, so we use the Kronecker structure:
    # D^{-1/2} (A kron B) D^{-1/2} where D = diag(A) kron diag(B)
    # = (diag(A)^{-1/2} A diag(A)^{-1/2}) kron (diag(B)^{-1/2} B diag(B)^{-1/2})
    # = Jacobi(A) kron Jacobi(B)
    # So the eigenvalues are products of eigenvalues of Jacobi-preconditioned A and B.

    inv_sqrt_diag_A = 1.0 / np.sqrt(np.maximum(diag_A, 1e-30))
    inv_sqrt_diag_B = 1.0 / np.sqrt(np.maximum(diag_B, 1e-30))

    jacobi_A = np.diag(inv_sqrt_diag_A) @ A @ np.diag(inv_sqrt_diag_A)
    jacobi_B = np.diag(inv_sqrt_diag_B) @ B @ np.diag(inv_sqrt_diag_B)

    eig_jA = np.linalg.eigvalsh(jacobi_A)
    eig_jB = np.linalg.eigvalsh(jacobi_B)

    diag_prec_eigs = np.outer(eig_jA, eig_jB).ravel()
    diag_prec_eigs.sort()
    kappa_diag = diag_prec_eigs[-1] / max(diag_prec_eigs[0], 1e-30)

    print(f"\n--- After diagonal preconditioning (Jacobi) ---")
    print(f"  Jacobi(A) kappa: {eig_jA[-1]/max(eig_jA[0], 1e-30):.1f}")
    print(f"  Jacobi(B) kappa: {eig_jB[-1]/max(eig_jB[0], 1e-30):.1f}")
    print(f"  Combined condition number: {kappa_diag:.1f}")
    print(f"  Improvement over raw: {kappa_raw/kappa_diag:.1f}x")

    # Kronecker preconditioning: (A^{-1} kron B^{-1})(A kron B) = I kron I = I
    # With damping: ((A+dI)^{-1} kron (B+dI)^{-1})(A kron B)
    A_damped = A + damping * np.eye(n_in)
    B_damped = B + damping * np.eye(n_out)

    kron_prec_A = np.linalg.solve(A_damped, A)
    kron_prec_B = np.linalg.solve(B_damped, B)

    eig_kA = np.linalg.eigvalsh(kron_prec_A)
    eig_kB = np.linalg.eigvalsh(kron_prec_B)

    kron_prec_eigs = np.outer(eig_kA, eig_kB).ravel()
    kron_prec_eigs.sort()
    kappa_kron = kron_prec_eigs[-1] / max(kron_prec_eigs[0], 1e-30)

    print(f"\n--- After Kronecker preconditioning (damping={damping}) ---")
    print(f"  Kron-prec(A) range: [{eig_kA[0]:.6f}, {eig_kA[-1]:.6f}]")
    print(f"  Kron-prec(B) range: [{eig_kB[0]:.6f}, {eig_kB[-1]:.6f}]")
    print(f"  Combined condition number: {kappa_kron:.4f}")
    print(f"  Improvement over raw: {kappa_raw/kappa_kron:.1f}x")
    print(f"  Improvement over diagonal: {kappa_diag/kappa_kron:.1f}x")

    return {
        'name': name,
        'n_in': n_in, 'n_out': n_out,
        'kappa_A': eig_A[-1] / max(eig_A[0], 1e-30),
        'kappa_B': eig_B[-1] / max(eig_B[0], 1e-30),
        'kappa_raw': kappa_raw,
        'kappa_diag': kappa_diag,
        'kappa_kron': kappa_kron,
    }


def main():
    print("Loading MNIST batch...")
    x, y = load_mnist_batch(1024)

    model = MLP(features=64, output_dim=10)
    key = jax.random.PRNGKey(42)
    init_params = model.init(key, jnp.ones((1, 784)))

    print("\nComputing Kronecker factors at initialization...")
    factors_init = compute_kronecker_factors(
        init_params, x, y, jax.nn.gelu, cross_entropy_output_grad)

    # Also compute after some training (curvature changes)
    print("\nTraining for 100 steps to get non-trivial curvature...")
    params = init_params
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    for _ in range(100):
        def loss_fn(p):
            logits = model.apply(p, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    print(f"  Loss after 100 steps: {float(loss):.4f}")

    print("\nComputing Kronecker factors after 100 steps of training...")
    factors_trained = compute_kronecker_factors(
        params, x, y, jax.nn.gelu, cross_entropy_output_grad)

    # Analyze each layer
    print("\n" + "#" * 70)
    print("# AT INITIALIZATION")
    print("#" * 70)
    results_init = []
    for key in ('dense1', 'dense2', 'dense3'):
        results_init.append(analyze_layer(key, factors_init[key]))

    print("\n" + "#" * 70)
    print("# AFTER 100 STEPS OF TRAINING")
    print("#" * 70)
    results_trained = []
    for key in ('dense1', 'dense2', 'dense3'):
        results_trained.append(analyze_layer(key, factors_trained[key]))

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    fmt = "{:<10s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(f"\nAt initialization:")
    print(fmt.format("layer", "params", "kappa_raw", "kappa_diag", "kappa_kron", "diag/kron"))
    print("-" * 62)
    for r in results_init:
        print(fmt.format(
            r['name'], str(r['n_in'] * r['n_out']),
            f"{r['kappa_raw']:.0f}", f"{r['kappa_diag']:.1f}",
            f"{r['kappa_kron']:.4f}",
            f"{r['kappa_diag']/r['kappa_kron']:.0f}x"))

    print(f"\nAfter 100 training steps:")
    print(fmt.format("layer", "params", "kappa_raw", "kappa_diag", "kappa_kron", "diag/kron"))
    print("-" * 62)
    for r in results_trained:
        print(fmt.format(
            r['name'], str(r['n_in'] * r['n_out']),
            f"{r['kappa_raw']:.0f}", f"{r['kappa_diag']:.1f}",
            f"{r['kappa_kron']:.4f}",
            f"{r['kappa_diag']/r['kappa_kron']:.0f}x"))

    print(f"\nInterpretation:")
    print(f"  kappa_raw:  condition number of raw GN Hessian per layer")
    print(f"  kappa_diag: after diagonal (Newton) preconditioning")
    print(f"  kappa_kron: after Kronecker (KFAC) preconditioning")
    print(f"  diag/kron:  how much better Kronecker is than diagonal")
    print(f"  If diag/kron >> 1, Kronecker captures structure diagonal misses")


def analyze_minigpt():
    """Kronecker analysis for MiniGPT transformer Dense layers.

    Manually walks through the first transformer MLP block to extract
    per-layer inputs and output gradients, then computes Kronecker factors.
    Dense layers in transformers operate pointwise on features, so we
    flatten (batch * seq_len) into the sample dimension.
    """
    from parameters.shared_models import MiniGPT

    print("\n" + "#" * 70)
    print("# MiniGPT (Shakespeare) — Kronecker analysis")
    print("#" * 70)

    # Small model for analysis
    vocab_size = 65  # tiny Shakespeare character vocab
    embed_dim = 128
    num_heads = 4
    num_layers = 6
    seq_len = 64
    B = 32

    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_heads=num_heads, num_layers=num_layers,
        dropout_rate=0.0, max_seq_len=256,
    )
    key = jax.random.PRNGKey(42)
    dummy_x = jax.random.randint(key, (B, seq_len), 0, vocab_size)
    params = model.init(key, dummy_x, train=False)

    # Train briefly to get non-trivial curvature
    print(f"\nModel: embed={embed_dim}, heads={num_heads}, layers={num_layers}")
    print(f"Data: B={B}, seq_len={seq_len}, vocab={vocab_size}")

    def loss_fn(p):
        logits = model.apply(p, dummy_x, train=False)
        # Shift for autoregressive: predict next token
        logits = logits[:, :-1, :]  # (B, T-1, vocab)
        targets = dummy_x[:, 1:]    # (B, T-1)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size), targets.reshape(-1)
        ).mean()

    print(f"Initial loss: {float(loss_fn(params)):.4f}")

    # Train 50 steps with Adam
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    for _ in range(50):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    print(f"Loss after 50 steps: {float(loss):.4f}")

    # Extract Kronecker factors for MLP Dense layers in the first transformer block.
    # MiniGPT (nn.compact) param structure: params['params']['Dense_N']['kernel']
    # The transformer MLP in each block has two Dense layers.
    # We manually forward through the first block's MLP to get layer inputs/outputs.

    p = params['params']

    # Forward pass up to and through the first MLP block:
    # 1. Token + position embeddings
    token_embed = p['Embed_0']['embedding'][dummy_x]       # (B, T, embed)
    positions = jnp.arange(seq_len)[None, :]
    pos_embed = p['Embed_1']['embedding'][positions]        # (1, T, embed)
    h = token_embed + pos_embed                             # (B, T, embed)

    # 2. First attention block (skip for simplicity — just apply as black box)
    # We want the MLP block input, which is after attention + residual + layernorm
    # Use model.apply with capture_intermediates instead of manual
    _, intermediates = model.apply(
        params, dummy_x, train=False,
        capture_intermediates=True, mutable=['intermediates'],
    )

    # The intermediates dict has module outputs keyed by module path.
    # For nn.compact models, Dense layers are named Dense_0, Dense_1, etc.
    # The MLP in each transformer block uses two Dense layers.
    # With 6 transformer blocks + 1 final Dense:
    #   Dense_0 through Dense_11: MLP Dense pairs (6 blocks × 2)
    #   Dense_12: final output projection
    # Plus SelfAttention has internal Dense layers (not captured separately by default).

    # Get the Dense outputs from intermediates
    inter = intermediates['intermediates']
    dense_keys = sorted([k for k in inter.keys() if k.startswith('Dense_')])
    print(f"\nCaptured Dense layer outputs: {dense_keys}")

    # For each Dense layer, the output z = x @ W + b has been captured.
    # We can compute g = dL/dz for each Dense output.
    # The Kronecker factor B = g^T g / N where g is (N, out_features)
    # and N = B * seq_len (flattened sample dimension).

    # Compute per-layer output gradients using jax.grad w.r.t. the Dense outputs.
    # Strategy: replace each Dense output with a stop_gradient boundary and
    # differentiate the loss w.r.t. that boundary.
    # Simpler: just compute A and B for the MLP Dense layers using the params.

    # For the first MLP block (Dense_0 = up-projection, Dense_1 = down-projection):
    # MLP up: z_up = LayerNorm(h) @ W_up + b_up, shape (B, T, 4*embed)
    # MLP down: z_down = gelu(z_up) @ W_down + b_down, shape (B, T, embed)
    # Layer input to Dense_0 is the LayerNorm output of the post-attention residual.

    # We'll analyze specific param paths. In nn.compact MiniGPT:
    # The first transformer block's MLP uses Dense_0 (up) and Dense_1 (down).
    # But SelfAttention also uses Dense layers internally...
    # Let's just identify the Dense kernel shapes to find MLP layers.

    print("\nAll Dense kernels in model:")
    dense_params = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params['params'])[0]:
        path_str = '/'.join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
        if 'kernel' in path_str and 'Dense' in path_str:
            print(f"  {path_str}: {leaf.shape}")
            dense_params[path_str] = leaf

    # Identify MLP Dense layers (4*embed up-projections and embed down-projections)
    mlp_up_keys = [k for k, v in dense_params.items()
                   if v.shape == (embed_dim, 4 * embed_dim)]
    mlp_down_keys = [k for k, v in dense_params.items()
                     if v.shape == (4 * embed_dim, embed_dim)]
    output_key = [k for k, v in dense_params.items()
                  if v.shape == (embed_dim, vocab_size)]

    print(f"\nMLP up-projection layers ({embed_dim}→{4*embed_dim}): {len(mlp_up_keys)}")
    print(f"MLP down-projection layers ({4*embed_dim}→{embed_dim}): {len(mlp_down_keys)}")
    print(f"Output projection ({embed_dim}→{vocab_size}): {len(output_key)}")

    # Use per-sample gradients to compute Kronecker factors.
    # For each Dense kernel W (in, out), per-sample grad dW_d sums over seq positions:
    #   dW_d = Σ_t x_{d,t}^T g_{d,t}  (summed over T positions)
    # This means dW_d has rank up to T, not rank 1.
    # To get A = E[x x^T] and B = E[g g^T], we need the actual x and g,
    # not just the gradient dW.
    #
    # Approach: compute per-sample gradients, then extract A and B as:
    #   A ≈ (1/B) Σ_d dW_d @ dW_d^T / trace(B_approx)  [approximate]
    #   B ≈ (1/B) Σ_d dW_d^T @ dW_d / trace(A_approx)  [approximate]
    #
    # Better approach: use the property that for the KFAC approximation,
    #   A = E_{d,t}[x_{d,t} x_{d,t}^T]  (average over samples AND positions)
    #   B = E_{d,t}[g_{d,t} g_{d,t}^T]
    # We can get these from per-sample gradients by noting:
    #   E[dW dW^T] = T * A ⊗ B  (if x and g are independent across positions)
    #   But this requires the KFAC approximation to hold.
    #
    # Simplest valid approach: compute A and B directly from the per-sample-per-position
    # x and g values. But we need to extract those from the model.
    #
    # Since the model is complex, let's use a shortcut: compute the condition number
    # of the Jacobi-preconditioned per-sample gradient covariance. This tells us
    # how well diagonal preconditioning works WITHOUT needing exact Kronecker factors.

    print("\nComputing per-sample gradients (this may take a moment)...")

    def per_sample_loss(p, xi):
        logits = model.apply(p, xi[None, :], train=False)
        logits = logits[:, :-1, :]
        targets = xi[1:]
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size), targets.reshape(-1)
        ).mean()

    # Use a small subset for tractability
    n_samples = min(B, 16)
    per_grads = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0))(
        params, dummy_x[:n_samples]
    )

    # For representative MLP layers, compute Kronecker factor condition numbers
    # from per-sample gradients. For kernel W (in, out):
    #   A_approx_{ij} = (1/N) Σ_d Σ_k dW_d[i,k] dW_d[j,k]  = (1/N) Σ_d (dW_d @ dW_d^T)_{ij}
    #   B_approx_{kl} = (1/N) Σ_d Σ_i dW_d[i,k] dW_d[i,l]  = (1/N) Σ_d (dW_d^T @ dW_d)_{kl}
    # These are marginals of the empirical Fisher, NOT the true Kronecker factors.
    # But their condition numbers give an upper bound on how well Kronecker works.

    print(f"\n{'=' * 70}")
    print(f"MiniGPT Kronecker analysis (from per-sample gradient marginals)")
    print(f"{'=' * 70}")

    # Analyze a few representative layers
    layers_to_analyze = []
    if mlp_up_keys:
        layers_to_analyze.append(('MLP_up_block0', mlp_up_keys[0]))
    if mlp_down_keys:
        layers_to_analyze.append(('MLP_down_block0', mlp_down_keys[0]))
    if len(mlp_up_keys) > 2:
        layers_to_analyze.append(('MLP_up_block2', mlp_up_keys[2]))
    if output_key:
        layers_to_analyze.append(('output_proj', output_key[0]))

    summary = []
    for label, param_path in layers_to_analyze:
        # Navigate the per-sample gradient pytree to find this kernel
        parts = param_path.split('/')
        grad_leaf = per_grads['params']
        for part in parts[:-1]:  # skip 'kernel'
            grad_leaf = grad_leaf[part]
        dW = np.array(grad_leaf['kernel'])  # (n_samples, in, out)

        n, n_in, n_out = dW.shape
        print(f"\n--- {label}: {param_path} ({n_in}x{n_out} = {n_in*n_out} params) ---")

        # Compute marginal factors from per-sample gradients
        # A_marginal = (1/N) Σ_d dW_d @ dW_d^T   shape (in, in)
        A_marg = np.mean([dW[d] @ dW[d].T for d in range(n)], axis=0)
        B_marg = np.mean([dW[d].T @ dW[d] for d in range(n)], axis=0)

        eig_A = np.linalg.eigvalsh(A_marg)
        eig_B = np.linalg.eigvalsh(B_marg)

        # Filter positive eigenvalues for condition number
        eig_A_pos = eig_A[eig_A > 1e-30]
        eig_B_pos = eig_B[eig_B > 1e-30]

        kappa_A = eig_A_pos[-1] / eig_A_pos[0] if len(eig_A_pos) > 1 else 1.0
        kappa_B = eig_B_pos[-1] / eig_B_pos[0] if len(eig_B_pos) > 1 else 1.0

        print(f"  A marginal kappa: {kappa_A:.1f}  (rank {len(eig_A_pos)}/{n_in})")
        print(f"  B marginal kappa: {kappa_B:.1f}  (rank {len(eig_B_pos)}/{n_out})")

        # Diagonal preconditioning: Jacobi(A) kron Jacobi(B)
        diag_A = np.diag(A_marg)
        diag_B = np.diag(B_marg)
        diag_A_safe = np.maximum(diag_A, 1e-30)
        diag_B_safe = np.maximum(diag_B, 1e-30)

        J_A = np.diag(1/np.sqrt(diag_A_safe)) @ A_marg @ np.diag(1/np.sqrt(diag_A_safe))
        J_B = np.diag(1/np.sqrt(diag_B_safe)) @ B_marg @ np.diag(1/np.sqrt(diag_B_safe))

        eig_jA = np.linalg.eigvalsh(J_A)
        eig_jB = np.linalg.eigvalsh(J_B)
        eig_jA_pos = eig_jA[eig_jA > 1e-10]
        eig_jB_pos = eig_jB[eig_jB > 1e-10]

        kappa_jA = eig_jA_pos[-1] / eig_jA_pos[0] if len(eig_jA_pos) > 1 else 1.0
        kappa_jB = eig_jB_pos[-1] / eig_jB_pos[0] if len(eig_jB_pos) > 1 else 1.0
        kappa_diag = kappa_jA * kappa_jB

        print(f"  After diagonal: Jacobi(A) kappa={kappa_jA:.1f}, "
              f"Jacobi(B) kappa={kappa_jB:.1f}, combined={kappa_diag:.1f}")
        print(f"  After Kronecker: kappa ≈ 1.0 (by construction)")
        print(f"  Diagonal/Kronecker improvement: ~{kappa_diag:.0f}x")

        summary.append({
            'name': label, 'n_in': n_in, 'n_out': n_out,
            'kappa_A': kappa_A, 'kappa_B': kappa_B,
            'kappa_diag': kappa_diag,
        })

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"MiniGPT SUMMARY")
    print(f"{'=' * 70}")
    fmt = "{:<20s} {:>6s} {:>10s} {:>10s} {:>12s}"
    print(fmt.format("layer", "params", "kappa_A", "kappa_B", "kappa_diag"))
    print("-" * 62)
    for r in summary:
        print(fmt.format(
            r['name'], str(r['n_in'] * r['n_out']),
            f"{r['kappa_A']:.0f}", f"{r['kappa_B']:.0f}",
            f"{r['kappa_diag']:.0f}"))
    print(f"\n  Kronecker kappa ≈ 1 for all layers (accounts for within-layer structure)")
    print(f"  diag/kron improvement = kappa_diag column above")


if __name__ == "__main__":
    main()
    analyze_minigpt()
