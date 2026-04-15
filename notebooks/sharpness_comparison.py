#!/usr/bin/env python3
"""
Sharpness comparison: does the learnable diagonal metric act as implicit SAM?

Compares the Hessian sharpness (max eigenvalue, trace, top-k eigenvalues) of minima
found by SGD, Adam, and a minimal learn_diag induced-metric optimizer on a small
2-layer MLP + synthetic classification task.

Measures sharpness along trajectories and at matched loss levels to ensure fair
comparison. Two datasets: an easy one and a harder one with class overlap.

Usage:
    python analysis/sharpness_comparison.py
"""

import time
import json
from functools import partial
from typing import List, Dict, Tuple, Any

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from scipy.linalg import eigvalsh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_SAMPLES = 200
INPUT_DIM = 8
HIDDEN_DIM = 16
N_CLASSES = 4
N_STEPS = 3000
SHARPNESS_EVERY = 50  # compute sharpness along trajectory every N steps

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_dataset(key, n=N_SAMPLES, d=INPUT_DIM, c=N_CLASSES, noise=0.5, separation=2.0):
    """Generate a synthetic classification dataset with clustered Gaussians."""
    k1, k2, k3 = jax.random.split(key, 3)
    centres = jax.random.normal(k1, (c, d)) * separation
    labels = jax.random.randint(k2, (n,), 0, c)
    noise_arr = jax.random.normal(k3, (n, d)) * noise
    x = centres[labels] + noise_arr
    return x, labels

# ---------------------------------------------------------------------------
# 2-layer MLP (flat parameter vector for Hessian computation)
# ---------------------------------------------------------------------------

def init_params(key, d_in=INPUT_DIM, d_h=HIDDEN_DIM, d_out=N_CLASSES):
    k1, k2 = jax.random.split(key)
    scale1 = jnp.sqrt(2.0 / d_in)
    scale2 = jnp.sqrt(2.0 / d_h)
    W1 = jax.random.normal(k1, (d_in, d_h)) * scale1
    b1 = jnp.zeros(d_h)
    W2 = jax.random.normal(k2, (d_h, d_out)) * scale2
    b2 = jnp.zeros(d_out)
    flat, unravel_fn = jax.flatten_util.ravel_pytree((W1, b1, W2, b2))
    return flat, unravel_fn

def forward(flat_params, x, unravel_fn):
    W1, b1, W2, b2 = unravel_fn(flat_params)
    h = jnp.maximum(x @ W1 + b1, 0.0)
    logits = h @ W2 + b2
    return logits

def loss_fn(flat_params, x, y, unravel_fn):
    logits = forward(flat_params, x, unravel_fn)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])

def accuracy_fn(flat_params, x, y, unravel_fn):
    logits = forward(flat_params, x, unravel_fn)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)

# ---------------------------------------------------------------------------
# Hessian sharpness computation
# ---------------------------------------------------------------------------

def compute_sharpness(flat_params, x, y, unravel_fn, top_k=5):
    """Compute full Hessian and return sharpness metrics."""
    loss_at_params = partial(loss_fn, x=x, y=y, unravel_fn=unravel_fn)
    loss_val = float(loss_at_params(flat_params))
    acc = float(accuracy_fn(flat_params, x, y, unravel_fn))

    H = jax.hessian(loss_at_params)(flat_params)
    H_np = np.array(H)
    H_np = 0.5 * (H_np + H_np.T)

    eigs = eigvalsh(H_np)
    max_eig = float(eigs[-1])
    trace_h = float(np.sum(eigs))
    top_k_eigs = [float(e) for e in eigs[-top_k:][::-1]]
    # Also useful: number of negative eigenvalues (indicates saddle)
    n_negative = int(np.sum(eigs < -1e-8))

    return {
        "loss": loss_val,
        "accuracy": acc,
        "max_eigenvalue": max_eig,
        "trace_hessian": trace_h,
        "top_5_eigenvalues": top_k_eigs,
        "n_negative_eigenvalues": n_negative,
    }

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def train_sgd(params0, x, y, unravel_fn, lr, momentum=0.9, n_steps=N_STEPS):
    """Train with plain SGD + momentum."""
    grad_fn = jax.jit(jax.grad(partial(loss_fn, x=x, y=y, unravel_fn=unravel_fn)))
    theta = params0.copy()
    v = jnp.zeros_like(theta)
    trajectory = []

    for step in range(n_steps):
        g = grad_fn(theta)
        v = momentum * v + g
        theta = theta - lr * v

        if step % SHARPNESS_EVERY == 0 or step == n_steps - 1:
            sharp = compute_sharpness(theta, x, y, unravel_fn)
            sharp["step"] = step
            trajectory.append(sharp)

    return theta, trajectory

def train_adam(params0, x, y, unravel_fn, lr, beta1=0.9, beta2=0.999, eps=1e-8,
               n_steps=N_STEPS):
    """Train with Adam."""
    grad_fn = jax.jit(jax.grad(partial(loss_fn, x=x, y=y, unravel_fn=unravel_fn)))
    theta = params0.copy()
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    trajectory = []

    for step in range(1, n_steps + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)
        theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        if (step - 1) % SHARPNESS_EVERY == 0 or step == n_steps:
            sharp = compute_sharpness(theta, x, y, unravel_fn)
            sharp["step"] = step - 1
            trajectory.append(sharp)

    return theta, trajectory

def train_learn_diag(params0, x, y, unravel_fn,
                     lr=0.01, momentum=0.9, xi=1.0,
                     metric_lr=0.1, metric_reg=0.01, metric_clip=4.0,
                     n_steps=N_STEPS):
    """Train with the learnable diagonal metric optimizer."""
    grad_fn = jax.jit(jax.grad(partial(loss_fn, x=x, y=y, unravel_fn=unravel_fn)))
    theta = params0.copy()
    s = jnp.zeros_like(theta)
    mom = jnp.zeros_like(theta)
    trajectory = []

    for step in range(1, n_steps + 1):
        g = grad_fn(theta)

        # Momentum with bias correction
        mom = momentum * mom + (1 - momentum) * g
        mom_corr = mom / (1 - momentum**step)

        # Preconditioned gradient
        exp_s = jnp.exp(s)
        g_scaled = exp_s * mom_corr

        # Trace statistic for damping
        v = xi * jnp.sum(g * (exp_s * g))
        r = 1.0 / (1.0 + jnp.abs(v))

        # Parameter update
        theta = theta - lr * r * g_scaled

        # Metric update
        s = s + metric_lr * xi * exp_s * (g ** 2) - metric_lr * metric_reg * s
        s = s - jnp.mean(s)  # mean-centre
        s = jnp.clip(s, -metric_clip, metric_clip)

        if (step - 1) % SHARPNESS_EVERY == 0 or step == n_steps:
            sharp = compute_sharpness(theta, x, y, unravel_fn)
            sharp["step"] = step - 1
            sharp["s_std"] = float(jnp.std(s))
            sharp["s_max"] = float(jnp.max(s))
            sharp["s_min"] = float(jnp.min(s))
            sharp["damping_r"] = float(r)
            trajectory.append(sharp)

    return theta, trajectory, s

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_at_loss_level(trajectory, target_loss, tol=0.1):
    """Find the first trajectory point where loss <= target_loss.
    Returns None if never reached."""
    for t in trajectory:
        if t["loss"] <= target_loss:
            return t
    return None

def print_trajectory_table(trajs, name_order, every_n=4):
    """Print a compact trajectory comparison table."""
    # Use steps from the first trajectory
    ref_steps = [t["step"] for t in trajs[name_order[0]]]
    # Subsample
    selected = ref_steps[::every_n]
    if ref_steps[-1] not in selected:
        selected.append(ref_steps[-1])

    step_dicts = {}
    for name in name_order:
        step_dicts[name] = {t["step"]: t for t in trajs[name]}

    print(f"  {'Step':>6}", end="")
    for name in name_order:
        print(f"  {name:>14}", end="")
    print()

    for step in selected:
        print(f"  {step:6d}", end="")
        for name in name_order:
            d = step_dicts[name]
            if step in d:
                print(f"  {d[step]['max_eigenvalue']:14.4f}", end="")
            else:
                print(f"  {'---':>14}", end="")
        print()

# ---------------------------------------------------------------------------
# Run one experimental setting
# ---------------------------------------------------------------------------

def run_experiment(label, x, y, params0, unravel_fn,
                   sgd_lr, adam_lr, ld_lr, ld_xi, ld_metric_lr,
                   ld_metric_reg, ld_metric_clip, n_steps):
    """Run all three optimizers on the given dataset and return results."""

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*80}")

    n_params = params0.shape[0]
    print(f"  Dataset: {x.shape[0]} samples, {x.shape[1]} features, {N_CLASSES} classes")
    print(f"  Model: MLP ({INPUT_DIM} -> {HIDDEN_DIM} -> {N_CLASSES}), {n_params} params")

    # Initial sharpness
    init_sharp = compute_sharpness(params0, x, y, unravel_fn)
    print(f"  Init: loss={init_sharp['loss']:.4f}  max_eig={init_sharp['max_eigenvalue']:.4f}  "
          f"trace={init_sharp['trace_hessian']:.4f}")

    # --- Train ---
    print(f"\n  Training SGD (lr={sgd_lr}) ...")
    t0 = time.time()
    sgd_params, sgd_traj = train_sgd(params0, x, y, unravel_fn, lr=sgd_lr, n_steps=n_steps)
    print(f"    done in {time.time()-t0:.1f}s, final loss={sgd_traj[-1]['loss']:.6f}")

    print(f"  Training Adam (lr={adam_lr}) ...")
    t0 = time.time()
    adam_params, adam_traj = train_adam(params0, x, y, unravel_fn, lr=adam_lr, n_steps=n_steps)
    print(f"    done in {time.time()-t0:.1f}s, final loss={adam_traj[-1]['loss']:.6f}")

    print(f"  Training Learn-Diag (lr={ld_lr}, xi={ld_xi}, metric_lr={ld_metric_lr}) ...")
    t0 = time.time()
    ld_params, ld_traj, final_s = train_learn_diag(
        params0, x, y, unravel_fn,
        lr=ld_lr, xi=ld_xi, metric_lr=ld_metric_lr,
        metric_reg=ld_metric_reg, metric_clip=ld_metric_clip, n_steps=n_steps
    )
    print(f"    done in {time.time()-t0:.1f}s, final loss={ld_traj[-1]['loss']:.6f}")

    # --- Final results table ---
    results = {"SGD": sgd_traj[-1], "Adam": adam_traj[-1], "Learn-Diag": ld_traj[-1]}
    trajs = {"SGD": sgd_traj, "Adam": adam_traj, "Learn-Diag": ld_traj}
    names = ["SGD", "Adam", "Learn-Diag"]

    print(f"\n  --- Final point comparison ---")
    print(f"  {'Optimizer':<14} {'Loss':>10} {'Acc':>6} {'Max Eig':>10} {'Trace(H)':>10}"
          f"  Top-5 Eigenvalues")
    print(f"  " + "-" * 88)
    for name in names:
        r = results[name]
        top5 = "  ".join(f"{e:8.4f}" for e in r["top_5_eigenvalues"])
        print(f"  {name:<14} {r['loss']:10.6f} {r['accuracy']:6.3f} "
              f"{r['max_eigenvalue']:10.4f} {r['trace_hessian']:10.4f}  {top5}")

    # --- Matched loss comparison ---
    # Find the worst (highest) final loss among all three
    all_final_losses = [results[n]["loss"] for n in names]
    max_final_loss = max(all_final_losses)
    # Try several loss thresholds
    thresholds = [0.1, 0.05, 0.01, 0.005, max_final_loss * 2]
    thresholds = sorted(set([t for t in thresholds if t > max_final_loss * 0.5]), reverse=True)

    print(f"\n  --- Matched-loss comparison ---")
    print(f"  (Sharpness when each optimizer first reaches a given loss level)")
    for threshold in thresholds:
        matched = {}
        all_found = True
        for name in names:
            m = find_at_loss_level(trajs[name], threshold)
            if m is None:
                all_found = False
                break
            matched[name] = m

        if all_found:
            print(f"\n  At loss <= {threshold:.4f}:")
            print(f"    {'Optimizer':<14} {'Step':>6} {'Loss':>10} {'Max Eig':>10} {'Trace':>10}")
            for name in names:
                m = matched[name]
                print(f"    {name:<14} {m['step']:6d} {m['loss']:10.6f} "
                      f"{m['max_eigenvalue']:10.4f} {m['trace_hessian']:10.4f}")

    # --- Trajectory table ---
    print(f"\n  --- Trajectory max eigenvalue (subsampled) ---")
    print_trajectory_table(trajs, names)

    # --- Trajectory statistics ---
    print(f"\n  --- Trajectory summary ---")
    for name in names:
        traj = trajs[name]
        max_eigs = [t["max_eigenvalue"] for t in traj]
        losses = [t["loss"] for t in traj]
        # Only look at post-convergence (loss < 0.1)
        converged = [(t["max_eigenvalue"], t["trace_hessian"]) for t in traj if t["loss"] < 0.1]
        if converged:
            conv_max_eigs = [c[0] for c in converged]
            conv_traces = [c[1] for c in converged]
            print(f"    {name:<14} post-convergence avg max_eig: {np.mean(conv_max_eigs):.4f}  "
                  f"avg trace: {np.mean(conv_traces):.4f}  "
                  f"min max_eig: {np.min(conv_max_eigs):.4f}")
        else:
            print(f"    {name:<14} never reached loss < 0.1")

    # --- Learn-Diag metric diagnostics ---
    if "s_std" in ld_traj[-1]:
        print(f"\n  --- Learn-Diag metric diagnostics ---")
        print(f"    Final s std:   {ld_traj[-1]['s_std']:.4f}")
        print(f"    Final s range: [{ld_traj[-1]['s_min']:.4f}, {ld_traj[-1]['s_max']:.4f}]")
        print(f"    Final damping: r={ld_traj[-1]['damping_r']:.6f}")

    return {
        "label": label,
        "initial_sharpness": init_sharp,
        "final_results": results,
        "trajectories": trajs,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("SHARPNESS COMPARISON: Does Learn-Diag Act as Implicit SAM?")
    print("=" * 80)

    key = jax.random.PRNGKey(SEED)
    all_results = []

    # -----------------------------------------------------------------------
    # Experiment 1: Easy problem (well-separated clusters)
    # -----------------------------------------------------------------------
    k1, k2, key = jax.random.split(key, 3)
    x_easy, y_easy = make_dataset(k1, noise=0.5, separation=2.0)
    params0_easy, unravel_easy = init_params(k2)

    res1 = run_experiment(
        label="Easy (separation=2.0, noise=0.5)",
        x=x_easy, y=y_easy, params0=params0_easy, unravel_fn=unravel_easy,
        sgd_lr=0.005,  # slower SGD so it doesn't trivially overshoot
        adam_lr=0.001,
        ld_lr=0.01, ld_xi=1.0, ld_metric_lr=0.1,
        ld_metric_reg=0.01, ld_metric_clip=4.0,
        n_steps=3000,
    )
    all_results.append(res1)

    # -----------------------------------------------------------------------
    # Experiment 2: Hard problem (overlapping clusters, more noise)
    # -----------------------------------------------------------------------
    k1, k2, key = jax.random.split(key, 3)
    x_hard, y_hard = make_dataset(k1, noise=1.5, separation=1.0)
    params0_hard, unravel_hard = init_params(k2)

    res2 = run_experiment(
        label="Hard (separation=1.0, noise=1.5)",
        x=x_hard, y=y_hard, params0=params0_hard, unravel_fn=unravel_hard,
        sgd_lr=0.005,
        adam_lr=0.001,
        ld_lr=0.01, ld_xi=1.0, ld_metric_lr=0.1,
        ld_metric_reg=0.01, ld_metric_clip=4.0,
        n_steps=3000,
    )
    all_results.append(res2)

    # -----------------------------------------------------------------------
    # Experiment 3: Same as hard, but with higher xi (stronger metric effect)
    # -----------------------------------------------------------------------
    k1, k2, key = jax.random.split(key, 3)
    # Re-use the hard dataset but new params init
    params0_3, unravel_3 = init_params(k2)

    res3 = run_experiment(
        label="Hard + high xi (xi=5.0, metric_lr=0.05)",
        x=x_hard, y=y_hard, params0=params0_3, unravel_fn=unravel_3,
        sgd_lr=0.005,
        adam_lr=0.001,
        ld_lr=0.01, ld_xi=5.0, ld_metric_lr=0.05,
        ld_metric_reg=0.01, ld_metric_clip=4.0,
        n_steps=3000,
    )
    all_results.append(res3)

    # -----------------------------------------------------------------------
    # Summary across experiments
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CROSS-EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\n{'Experiment':<40} {'Opt':<14} {'Loss':>10} {'Max Eig':>10} {'Trace':>10}")
    print("-" * 90)
    for res in all_results:
        for name in ["SGD", "Adam", "Learn-Diag"]:
            r = res["final_results"][name]
            prefix = res["label"] if name == "SGD" else ""
            print(f"{prefix:<40} {name:<14} {r['loss']:10.6f} "
                  f"{r['max_eigenvalue']:10.4f} {r['trace_hessian']:10.4f}")
        print()

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    for res in all_results:
        r = res["final_results"]
        trajs_exp = res["trajectories"]
        sgd_max = r["SGD"]["max_eigenvalue"]
        adam_max = r["Adam"]["max_eigenvalue"]
        ld_max = r["Learn-Diag"]["max_eigenvalue"]
        sgd_trace = r["SGD"]["trace_hessian"]
        adam_trace = r["Adam"]["trace_hessian"]
        ld_trace = r["Learn-Diag"]["trace_hessian"]

        print(f"\n  {res['label']}:")

        # Compare at similar loss levels for fairness
        sgd_loss = r["SGD"]["loss"]
        adam_loss = r["Adam"]["loss"]
        ld_loss = r["Learn-Diag"]["loss"]

        if abs(adam_loss - ld_loss) / max(adam_loss, ld_loss, 1e-10) < 0.5:
            # Similar loss -- direct comparison is fair
            if ld_max < adam_max:
                pct = (1 - ld_max / adam_max) * 100
                print(f"    Learn-Diag is {pct:.1f}% flatter than Adam (max eigenvalue)")
                print(f"    at comparable loss ({ld_loss:.6f} vs {adam_loss:.6f})")
            else:
                pct = (ld_max / adam_max - 1) * 100
                print(f"    Learn-Diag is {pct:.1f}% SHARPER than Adam (max eigenvalue)")
        else:
            print(f"    Losses differ substantially (Adam={adam_loss:.6f}, LD={ld_loss:.6f})")
            print(f"    -- use matched-loss comparison above for fair comparison")

        if ld_trace < adam_trace:
            pct_t = (1 - ld_trace / adam_trace) * 100
            print(f"    Learn-Diag trace is {pct_t:.1f}% lower than Adam")

        # Matched-loss sharpness comparison (the key metric)
        # Find the loss level where all three converged
        print(f"\n    Matched-loss sharpness analysis:")
        for threshold in [0.5, 0.1, 0.05, 0.01]:
            matched = {}
            all_found = True
            for name in ["SGD", "Adam", "Learn-Diag"]:
                m = find_at_loss_level(trajs_exp[name], threshold)
                if m is None:
                    all_found = False
                    break
                matched[name] = m
            if all_found:
                ld_m = matched["Learn-Diag"]["max_eigenvalue"]
                adam_m = matched["Adam"]["max_eigenvalue"]
                sgd_m = matched["SGD"]["max_eigenvalue"]
                if ld_m < adam_m:
                    pct_vs_adam = (1 - ld_m / adam_m) * 100
                    print(f"      At loss<={threshold}: LD max_eig={ld_m:.4f} is "
                          f"{pct_vs_adam:.1f}% flatter than Adam ({adam_m:.4f})")
                else:
                    pct_vs_adam = (ld_m / adam_m - 1) * 100
                    print(f"      At loss<={threshold}: LD max_eig={ld_m:.4f} is "
                          f"{pct_vs_adam:.1f}% sharper than Adam ({adam_m:.4f})")
                if ld_m < sgd_m:
                    pct_vs_sgd = (1 - ld_m / sgd_m) * 100
                    print(f"        vs SGD ({sgd_m:.4f}): {pct_vs_sgd:.1f}% flatter")
                else:
                    pct_vs_sgd = (ld_m / sgd_m - 1) * 100
                    print(f"        vs SGD ({sgd_m:.4f}): {pct_vs_sgd:.1f}% sharper")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpness_results.json")

    # Make trajectories JSON-serializable (strip numpy types)
    def sanitize(obj):
        if isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    save_data = sanitize(all_results)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
