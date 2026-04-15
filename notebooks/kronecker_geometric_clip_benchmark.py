"""
Kronecker geometric clipping benchmark (IMO-153).

Compares plain KFAC vs KFAC + geometric clipping vs Adam on MNIST MLP.

The induced metric framework with Kronecker base metric gives:

    step = KFAC_step / (1 + xi * preconditioned_norm^2)

where:
    KFAC_step = A^{-1} @ grad_W @ B^{-1}  (per layer)
    preconditioned_norm^2 = sum over layers of tr(grad_W^T @ A^{-1} @ grad_W @ B^{-1})

The denominator provides smooth gradient clipping. Step norm bounded by 1/(2*sqrt(xi)).

Focus: stability/robustness to lr, early-training behavior, divergence prevention.

Run:
    PYTHONPATH=repo .conda/bin/python3.11 repo/notebooks/kronecker_geometric_clip_benchmark.py
"""

import sys
import os
import time
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.insert(0, repo_dir)

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from optimisers.jax_gn_diag import cross_entropy_output_grad
from optimisers.jax_kronecker import kronecker_factors_mlp, apply_kronecker_preconditioner
from parameters.shared_models import MLP


# ---- Data -------------------------------------------------------------------

def load_mnist(n_train=10000, n_test=2000):
    """Load MNIST with a subset for speed. Full-batch on n_train samples."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = jnp.array(x_train[:n_train].reshape(-1, 784) / 255.0, dtype=jnp.float32)
    x_test = jnp.array(x_test[:n_test].reshape(-1, 784) / 255.0, dtype=jnp.float32)
    y_train = jnp.array(y_train[:n_train])
    y_test = jnp.array(y_test[:n_test])
    return x_train, y_train, x_test, y_test


def _flush():
    sys.stdout.flush()


# ---- Loss / accuracy --------------------------------------------------------

def loss_fn(params, x, y, model):
    logits = model.apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def accuracy(params, x, y, model):
    logits = model.apply(params, x)
    return float(jnp.mean(jnp.argmax(logits, axis=1) == y))


# ---- Preconditioned norm computation ----------------------------------------

def preconditioned_norm_sq(grads, kron_factors, layer_keys):
    """Compute sum over layers of tr(grad_W^T A^{-1} grad_W B^{-1}).

    This is the preconditioned gradient norm squared used in the
    geometric clipping denominator.

    For biases: tr(grad_b^T B^{-1} grad_b) = grad_b @ B^{-1} @ grad_b.
    """
    total = 0.0
    for key in layer_keys:
        A_inv, B_inv = kron_factors[key]
        g_kernel = grads['params'][key]['kernel']
        g_bias = grads['params'][key]['bias']

        # Kernel: tr(G^T A^{-1} G B^{-1})
        # = sum of elementwise product of (A^{-1} G B^{-1}) and G
        precond_kernel = A_inv @ g_kernel @ B_inv
        total += jnp.sum(precond_kernel * g_kernel)

        # Bias: grad_b @ B^{-1} @ grad_b
        precond_bias = B_inv @ g_bias
        total += jnp.sum(precond_bias * g_bias)

    return total


# ---- Training functions ------------------------------------------------------

LAYER_KEYS = ('dense1', 'dense2', 'dense3')
EVAL_STEPS = {100, 200, 500, 999}  # Steps at which to record metrics


@jax.jit
def _kfac_step(params, grads, mom_state, step, kron_factors, lr, momentum,
               xi, use_clipping):
    """Single KFAC step with optional geometric clipping. JIT-compiled."""
    one_minus_mom = 1.0 - momentum

    # Momentum on raw gradients with bias correction
    new_mom = jax.tree.map(
        lambda m, g: momentum * m + one_minus_mom * g,
        mom_state, grads,
    )
    m_corr = jax.tree.map(
        lambda m: m / (1.0 - momentum ** step),
        new_mom,
    )

    # Apply Kronecker preconditioning
    updates = apply_kronecker_preconditioner(m_corr, kron_factors, LAYER_KEYS)

    # Compute preconditioned norm for clipping factor
    norm_sq = preconditioned_norm_sq(m_corr, kron_factors, LAYER_KEYS)
    clip_factor = jnp.where(use_clipping, 1.0 / (1.0 + xi * norm_sq), 1.0)

    # Scale by -lr * clip_factor
    scale = -lr * clip_factor
    updates = jax.tree.map(lambda u: scale * u, updates)

    new_params = optax.apply_updates(params, updates)
    return new_params, new_mom, norm_sq, clip_factor


def train_kfac(params, model, x, y, x_test, y_test,
               lr, damping, xi, use_clipping, momentum,
               n_steps, label):
    """Train with KFAC, optionally with geometric clipping."""
    mom_state = jax.tree.map(jnp.zeros_like, params)
    trajectory = []
    clip_history = []
    t0 = time.time()
    diverged = False

    for step_idx in range(n_steps):
        step_num = jnp.array(step_idx + 1, dtype=jnp.float32)

        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
        loss_val = float(loss)

        # Detect divergence
        if math.isnan(loss_val) or math.isinf(loss_val) or loss_val > 100.0:
            diverged = True
            print(f"  [{label}] DIVERGED at step {step_idx}, loss={loss_val:.4f}")
            _flush()
            trajectory.append({
                'step': step_idx, 'loss': loss_val,
                'test_acc': 0.0, 'elapsed': time.time() - t0,
                'diverged': True,
            })
            break

        kf = kronecker_factors_mlp(params, x, y, jax.nn.gelu,
                                   cross_entropy_output_grad, damping=damping)

        params, mom_state, norm_sq, clip_factor = _kfac_step(
            params, grads, mom_state, step_num, kf, lr, momentum,
            xi, use_clipping,
        )

        clip_val = float(clip_factor)
        norm_sq_val = float(norm_sq)
        clip_history.append((step_idx, norm_sq_val, clip_val))

        eval_step = (step_idx % 50 == 0) or (step_idx in EVAL_STEPS) or (step_idx == n_steps - 1)
        if eval_step:
            acc = accuracy(params, x_test, y_test, model)
            elapsed = time.time() - t0
            trajectory.append({
                'step': step_idx, 'loss': loss_val,
                'test_acc': acc, 'elapsed': elapsed,
                'norm_sq': norm_sq_val, 'clip_factor': clip_val,
                'diverged': False,
            })
            if step_idx % 200 == 0 or step_idx in EVAL_STEPS:
                clip_str = f"  clip={clip_val:.4f}  ||s||^2={norm_sq_val:.2f}" if use_clipping else ""
                print(f"  [{label}] step={step_idx:4d}  loss={loss_val:.4f}  "
                      f"acc={acc:.4f}{clip_str}  ({elapsed:.1f}s)")
                _flush()

    return params, trajectory, clip_history, diverged


def train_adam(params, model, x, y, x_test, y_test,
               lr, n_steps, label):
    """Train with Adam."""
    opt = optax.adam(learning_rate=lr)
    state = opt.init(params)
    trajectory = []
    t0 = time.time()

    for step_idx in range(n_steps):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model))(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        loss_val = float(loss)
        eval_step = (step_idx % 50 == 0) or (step_idx in EVAL_STEPS) or (step_idx == n_steps - 1)
        if eval_step:
            acc = accuracy(params, x_test, y_test, model)
            elapsed = time.time() - t0
            trajectory.append({
                'step': step_idx, 'loss': loss_val,
                'test_acc': acc, 'elapsed': elapsed,
                'diverged': False,
            })
            if step_idx % 200 == 0 or step_idx in EVAL_STEPS:
                print(f"  [{label}] step={step_idx:4d}  loss={loss_val:.4f}  "
                      f"acc={acc:.4f}  ({elapsed:.1f}s)")
                _flush()

    return params, trajectory


# ---- Main --------------------------------------------------------------------

def main():
    print("=" * 80)
    print("KRONECKER GEOMETRIC CLIPPING BENCHMARK")
    print("=" * 80)
    print()
    print("MNIST MLP (784-64-64-10), full-batch (10K subset), 1000 steps")
    print("Geometric clipping: step = KFAC_step / (1 + xi * ||s||^2)")
    print()
    _flush()

    print("Loading MNIST (10K train, 2K test)...")
    _flush()
    x_train, y_train, x_test, y_test = load_mnist(n_train=10000, n_test=2000)

    model = MLP(features=64, output_dim=10)
    key = jax.random.PRNGKey(42)
    init_params = model.init(key, jnp.ones((1, 784)))

    n_steps = 1000
    x, y = x_train, y_train

    results = {}
    clip_histories = {}

    # ---- Adam baseline -------------------------------------------------------
    print("\n" + "-" * 60)
    print("ADAM BASELINE (lr=1e-3)")
    print("-" * 60)
    _flush()
    _, results['adam_lr=1e-3'] = train_adam(
        init_params, model, x, y, x_test, y_test,
        lr=1e-3, n_steps=n_steps, label="adam")

    # ---- Plain KFAC: lr sweep ------------------------------------------------
    kfac_lrs = [0.01, 0.05, 0.1, 0.5, 1.0]
    damping = 1e-4
    momentum = 0.9

    for lr in kfac_lrs:
        print(f"\n{'-' * 60}")
        print(f"PLAIN KFAC: lr={lr}, damping={damping}, momentum={momentum}")
        print(f"{'-' * 60}")
        _flush()
        key_name = f'kfac_lr={lr}'
        _, results[key_name], clip_histories[key_name], div = train_kfac(
            init_params, model, x, y, x_test, y_test,
            lr=lr, damping=damping, xi=0.0, use_clipping=False,
            momentum=momentum, n_steps=n_steps, label=f"kfac,lr={lr}")

    # ---- KFAC + geometric clipping: xi x lr sweep ---------------------------
    xi_values = [0.01, 0.1, 1.0, 10.0]

    for xi in xi_values:
        for lr in kfac_lrs:
            print(f"\n{'-' * 60}")
            print(f"KFAC+CLIP: lr={lr}, xi={xi}, damping={damping}, momentum={momentum}")
            print(f"  Max step norm = 1/(2*sqrt(xi)) = {1.0/(2.0*math.sqrt(xi)):.4f}")
            print(f"{'-' * 60}")
            _flush()
            key_name = f'clip_xi={xi}_lr={lr}'
            _, results[key_name], clip_histories[key_name], div = train_kfac(
                init_params, model, x, y, x_test, y_test,
                lr=lr, damping=damping, xi=xi, use_clipping=True,
                momentum=momentum, n_steps=n_steps,
                label=f"clip,xi={xi},lr={lr}")

    # ---- Results summary -----------------------------------------------------
    print("\n\n")
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    checkpoints = [100, 200, 500, 999]
    header = f"{'config':<35s}" + "".join(f"{'@'+str(s):>10s}" for s in checkpoints) + f"{'best':>10s}  {'diverged':>8s}"
    print(header)
    print("-" * len(header))

    # Helper to extract accuracy at checkpoint
    def acc_at_step(traj, target_step):
        acc_at = {t['step']: t for t in traj}
        if target_step in acc_at:
            return acc_at[target_step]['test_acc']
        # Find closest within 10 steps
        closest = min(acc_at.keys(), key=lambda k: abs(k - target_step)) if acc_at else None
        if closest is not None and abs(closest - target_step) <= 10:
            return acc_at[closest]['test_acc']
        return None

    def best_acc(traj):
        accs = [t['test_acc'] for t in traj if not t.get('diverged', False)]
        return max(accs) if accs else 0.0

    # Print Adam first
    for name in sorted(results.keys()):
        if name.startswith('adam'):
            traj = results[name]
            vals = []
            for s in checkpoints:
                a = acc_at_step(traj, s)
                vals.append(f"{a:.4f}" if a is not None else "N/A")
            ba = best_acc(traj)
            div = any(t.get('diverged', False) for t in traj)
            line = f"{name:<35s}" + "".join(f"{v:>10s}" for v in vals) + f"{ba:>10.4f}  {'YES' if div else 'no':>8s}"
            print(line)

    print()

    # Print plain KFAC
    for name in sorted(results.keys()):
        if name.startswith('kfac'):
            traj = results[name]
            vals = []
            for s in checkpoints:
                a = acc_at_step(traj, s)
                vals.append(f"{a:.4f}" if a is not None else "N/A")
            ba = best_acc(traj)
            div = any(t.get('diverged', False) for t in traj)
            line = f"{name:<35s}" + "".join(f"{v:>10s}" for v in vals) + f"{ba:>10.4f}  {'YES' if div else 'no':>8s}"
            print(line)

    print()

    # Print clipped, grouped by xi
    for xi in xi_values:
        for name in sorted(results.keys()):
            if name.startswith(f'clip_xi={xi}_'):
                traj = results[name]
                vals = []
                for s in checkpoints:
                    a = acc_at_step(traj, s)
                    vals.append(f"{a:.4f}" if a is not None else "N/A")
                ba = best_acc(traj)
                div = any(t.get('diverged', False) for t in traj)
                line = f"{name:<35s}" + "".join(f"{v:>10s}" for v in vals) + f"{ba:>10.4f}  {'YES' if div else 'no':>8s}"
                print(line)
        print()

    # ---- Divergence analysis -------------------------------------------------
    print("\n" + "=" * 100)
    print("DIVERGENCE ANALYSIS")
    print("=" * 100)

    print("\nPlain KFAC divergence by lr:")
    for lr in kfac_lrs:
        key = f'kfac_lr={lr}'
        traj = results[key]
        div = any(t.get('diverged', False) for t in traj)
        ba = best_acc(traj)
        print(f"  lr={lr:<6}  diverged={div}  best_acc={ba:.4f}")

    print("\nKFAC+clip divergence by (xi, lr):")
    for xi in xi_values:
        print(f"\n  xi={xi}:")
        for lr in kfac_lrs:
            key = f'clip_xi={xi}_lr={lr}'
            traj = results[key]
            div = any(t.get('diverged', False) for t in traj)
            ba = best_acc(traj)
            print(f"    lr={lr:<6}  diverged={div}  best_acc={ba:.4f}")

    # ---- Stable lr range -----------------------------------------------------
    print("\n" + "=" * 100)
    print("STABLE LR RANGE")
    print("=" * 100)

    def stable_lrs(prefix, xi=None):
        stable = []
        for lr in kfac_lrs:
            if xi is not None:
                key = f'clip_xi={xi}_lr={lr}'
            else:
                key = f'kfac_lr={lr}'
            traj = results[key]
            div = any(t.get('diverged', False) for t in traj)
            if not div:
                stable.append(lr)
        return stable

    plain_stable = stable_lrs('kfac')
    print(f"\nPlain KFAC stable lrs: {plain_stable}")
    if plain_stable:
        print(f"  Range: [{min(plain_stable)}, {max(plain_stable)}]")

    for xi in xi_values:
        clip_stable = stable_lrs('clip', xi=xi)
        print(f"\nKFAC+clip xi={xi} stable lrs: {clip_stable}")
        if clip_stable:
            print(f"  Range: [{min(clip_stable)}, {max(clip_stable)}]")
            print(f"  Max step norm bound: {1.0/(2.0*math.sqrt(xi)):.4f}")

    # ---- Clipping factor analysis --------------------------------------------
    print("\n" + "=" * 100)
    print("CLIPPING FACTOR ANALYSIS: 1/(1 + xi * ||s||^2)")
    print("=" * 100)

    for xi in xi_values:
        print(f"\nxi={xi}:")
        for lr in kfac_lrs:
            key = f'clip_xi={xi}_lr={lr}'
            ch = clip_histories.get(key, [])
            if not ch:
                print(f"  lr={lr}: no data (diverged immediately)")
                continue

            # Early phase (first 50 steps)
            early = [c for (s, n, c) in ch if s < 50]
            # Mid phase (steps 50-200)
            mid = [c for (s, n, c) in ch if 50 <= s < 200]
            # Late phase (steps 500+)
            late = [c for (s, n, c) in ch if s >= 500]

            def stats(vals):
                if not vals:
                    return "N/A"
                mn = min(vals)
                mx = max(vals)
                avg = sum(vals) / len(vals)
                return f"min={mn:.4f} max={mx:.4f} avg={avg:.4f}"

            print(f"  lr={lr}:")
            print(f"    Early (0-49):    {stats(early)}")
            print(f"    Mid   (50-199):  {stats(mid)}")
            print(f"    Late  (500+):    {stats(late)}")

            # Also show norm_sq stats
            early_n = [n for (s, n, c) in ch if s < 50]
            late_n = [n for (s, n, c) in ch if s >= 500]
            if early_n:
                print(f"    ||s||^2 early:   min={min(early_n):.2f} max={max(early_n):.2f} avg={sum(early_n)/len(early_n):.2f}")
            if late_n:
                print(f"    ||s||^2 late:    min={min(late_n):.2f} max={max(late_n):.2f} avg={sum(late_n)/len(late_n):.2f}")

    # ---- Best configs --------------------------------------------------------
    print("\n" + "=" * 100)
    print("BEST CONFIGS BY FINAL ACCURACY (@999)")
    print("=" * 100)

    final_accs = []
    for name, traj in results.items():
        a = acc_at_step(traj, 999)
        div = any(t.get('diverged', False) for t in traj)
        if a is not None and not div:
            final_accs.append((name, a))
    final_accs.sort(key=lambda x: -x[1])

    for i, (name, acc) in enumerate(final_accs[:15]):
        print(f"  {i+1:2d}. {name:<35s}  {acc:.4f}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
