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


if __name__ == "__main__":
    main()
