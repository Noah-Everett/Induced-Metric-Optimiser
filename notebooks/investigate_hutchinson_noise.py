"""
IMO-127: Investigate derived optimizer failures on small examples.

Runs sgd_newton_diag on 2D test functions in three modes:
  (a) Exact Hessian diagonal (no Hutchinson)
  (b) Single Hutchinson sample (current default)
  (c) Multi-sample Hutchinson (K=2, 5, 10)

Tracks per-step:
  - s_i trajectories
  - Fraction of negative Hutchinson estimates (eps-clamp events)
  - Loss convergence
  - Clipping fraction

Prints summary tables and writes CSV trajectories for plotting.

Run: PYTHONPATH=repo .conda/bin/python repo/notebooks/investigate_hutchinson_noise.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import csv
import os
import time

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Test functions (N=2)
# ---------------------------------------------------------------------------

def rosenbrock(theta):
    x, y = theta[0], theta[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def beale(theta):
    x, y = theta[0], theta[1]
    return ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2
            + (2.625 - x + x*y**3)**2)

def himmelblau(theta):
    x, y = theta[0], theta[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def ackley(theta):
    n = theta.shape[0]
    a, b, c = 20.0, 0.2, 2 * jnp.pi
    return (-a * jnp.exp(-b * jnp.sqrt(jnp.sum(theta**2) / n))
            - jnp.exp(jnp.sum(jnp.cos(c * theta)) / n) + a + jnp.e)

def saddle(theta):
    return theta[0]**2 - theta[1]**2


FUNCTIONS = {
    "Rosenbrock": (rosenbrock, jnp.array([-1.2, 1.0]),  1e-6),
    "Beale":      (beale,      jnp.array([0.5, 0.5]),   1e-6),
    "Himmelblau": (himmelblau, jnp.array([0.0, 0.0]),   1e-6),
    "Ackley":     (ackley,     jnp.array([0.5, 0.5]),   1e-4),
    "Saddle":     (saddle,     jnp.array([1.0, 1.0]),   None),
}


# ---------------------------------------------------------------------------
# Hessian diagonal estimators
# ---------------------------------------------------------------------------

def exact_hess_diag(loss_fn, theta):
    """Compute exact Hessian diagonal via full Hessian."""
    H = jax.hessian(loss_fn)(theta)
    return jnp.diag(H)


def hutchinson_diag_single(loss_fn, theta, key):
    """Single Hutchinson sample (current default)."""
    v = 2.0 * jax.random.bernoulli(key, 0.5, theta.shape).astype(theta.dtype) - 1.0
    _, hvp = jax.jvp(jax.grad(loss_fn), (theta,), (v,))
    return v * hvp


def hutchinson_diag_multi(loss_fn, theta, key, K):
    """Average of K Hutchinson samples."""
    keys = jax.random.split(key, K)
    def single(k):
        v = 2.0 * jax.random.bernoulli(k, 0.5, theta.shape).astype(theta.dtype) - 1.0
        _, hvp = jax.jvp(jax.grad(loss_fn), (theta,), (v,))
        return v * hvp
    estimates = jax.vmap(single)(keys)
    return jnp.mean(estimates, axis=0)


# ---------------------------------------------------------------------------
# Training step builders
# ---------------------------------------------------------------------------

def make_step_exact(loss_fn, lr, mom_coeff, beta_s, clip_val, hess_eps):
    """Step using exact Hessian diagonal."""
    neg_lr = -lr
    one_minus_mom = 1.0 - mom_coeff
    one_minus_bs = 1.0 - beta_s
    lo, hi = -abs(clip_val), abs(clip_val)

    @jax.jit
    def step(theta, s, m, step_count, key):
        step_count = step_count + 1
        g = jax.grad(loss_fn)(theta)
        h_diag = exact_hess_diag(loss_fn, theta)

        # Track diagnostics: how many components are negative
        n_neg = jnp.sum(h_diag < 0.0)

        # Metric EMA
        s_target = -jnp.log(jnp.maximum(h_diag, hess_eps))
        s_new = one_minus_bs * s + beta_s * s_target
        s_before_clip = s_new - jnp.mean(s_new)
        s_new = jnp.clip(s_before_clip, lo, hi)

        # Momentum + bias correction
        m_new = mom_coeff * m + one_minus_mom * g
        m_corr = m_new / (1.0 - mom_coeff ** step_count)

        # Preconditioned step
        theta_new = theta + neg_lr * jnp.exp(s_new) * m_corr
        loss = loss_fn(theta_new)

        # Clipping fraction
        n_clipped = jnp.sum(jnp.abs(s_before_clip) > abs(clip_val) - 1e-6)

        return theta_new, s_new, m_new, step_count, key, loss, h_diag, s_target, n_neg, n_clipped, s_before_clip

    return step


def make_step_hutchinson(loss_fn, lr, mom_coeff, beta_s, clip_val, hess_eps, K=1):
    """Step using Hutchinson diagonal estimate (K samples)."""
    neg_lr = -lr
    one_minus_mom = 1.0 - mom_coeff
    one_minus_bs = 1.0 - beta_s
    lo, hi = -abs(clip_val), abs(clip_val)

    @jax.jit
    def step(theta, s, m, step_count, key):
        step_count = step_count + 1
        g = jax.grad(loss_fn)(theta)

        key, subkey = jax.random.split(key)
        if K == 1:
            h_diag = hutchinson_diag_single(loss_fn, theta, subkey)
        else:
            h_diag = hutchinson_diag_multi(loss_fn, theta, subkey, K)

        n_neg = jnp.sum(h_diag < 0.0)

        s_target = -jnp.log(jnp.maximum(h_diag, hess_eps))
        s_new = one_minus_bs * s + beta_s * s_target
        s_before_clip = s_new - jnp.mean(s_new)
        s_new = jnp.clip(s_before_clip, lo, hi)

        m_new = mom_coeff * m + one_minus_mom * g
        m_corr = m_new / (1.0 - mom_coeff ** step_count)

        theta_new = theta + neg_lr * jnp.exp(s_new) * m_corr
        loss = loss_fn(theta_new)

        n_clipped = jnp.sum(jnp.abs(s_before_clip) > abs(clip_val) - 1e-6)

        return theta_new, s_new, m_new, step_count, key, loss, h_diag, s_target, n_neg, n_clipped, s_before_clip

    return step


# ---------------------------------------------------------------------------
# Runner with full diagnostics
# ---------------------------------------------------------------------------

def run_with_diagnostics(loss_fn, theta0, mode, clip_val, n_steps=2000,
                         lr=0.01, beta_s=0.01, mom=0.9, hess_eps=1e-8,
                         seed=42, K=1):
    """
    Run optimizer and collect per-step diagnostics.

    mode: "exact", "hutchinson", "multi-K"
    Returns dict with trajectories.
    """
    if mode == "exact":
        step_fn = make_step_exact(loss_fn, lr, mom, beta_s, clip_val, hess_eps)
    else:
        step_fn = make_step_hutchinson(loss_fn, lr, mom, beta_s, clip_val, hess_eps, K=K)

    theta = theta0.copy()
    s = jnp.zeros_like(theta)
    m = jnp.zeros_like(theta)
    sc = jnp.array(0, dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)

    # Warmup JIT
    step_fn(theta, s, m, sc, key)

    # Reset
    theta = theta0.copy()
    s = jnp.zeros_like(theta)
    m = jnp.zeros_like(theta)
    sc = jnp.array(0, dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)

    N = theta0.shape[0]
    history = {
        "loss": [],
        "s": [],          # s_i values per step (N-dim)
        "s_target": [],   # Newton target per step
        "h_diag": [],     # raw Hessian diag estimate
        "n_neg": [],      # number of negative h_diag components
        "n_clipped": [],  # number of s_i clipped
        "s_before_clip": [],
    }

    for t in range(n_steps):
        theta, s, m, sc, key, loss, h_diag, s_target, n_neg, n_clipped, s_bc = \
            step_fn(theta, s, m, sc, key)

        lv = float(loss)
        history["loss"].append(lv)
        history["s"].append(np.array(s))
        history["s_target"].append(np.array(s_target))
        history["h_diag"].append(np.array(h_diag))
        history["n_neg"].append(int(n_neg))
        history["n_clipped"].append(int(n_clipped))
        history["s_before_clip"].append(np.array(s_bc))

        if not np.isfinite(lv) or lv > 1e20:
            break

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Hyperparameter grid: (lr, clip_val) pairs to test
CONFIGS = [
    (0.001, 1.0),
    (0.001, 4.0),
    (0.005, 1.0),
    (0.005, 4.0),
    (0.01, 1.0),
    (0.01, 4.0),
]

MODES = [
    ("exact",      1),
    ("hutchinson", 1),
    ("multi-2",    2),
    ("multi-5",    5),
    ("multi-10",  10),
]

N_STEPS = 3000


def summarize(history, threshold):
    """Summarize a run."""
    losses = history["loss"]
    n = len(losses)
    final = losses[-1] if losses else float('inf')
    diverged = not np.isfinite(final) or final > 1e20

    steps_to_eps = None
    if threshold is not None and not diverged:
        for i, l in enumerate(losses):
            if l < threshold:
                steps_to_eps = i
                break

    total_neg = sum(history["n_neg"])
    total_steps = n
    neg_frac = total_neg / (total_steps * 2) if total_steps > 0 else 0  # N=2

    total_clipped = sum(history["n_clipped"])
    clip_frac = total_clipped / (total_steps * 2) if total_steps > 0 else 0

    s_final = history["s"][-1] if history["s"] else np.zeros(2)

    return {
        "final_loss": final,
        "steps": n,
        "steps_to_eps": steps_to_eps,
        "diverged": diverged,
        "neg_frac": neg_frac,
        "clip_frac": clip_frac,
        "s_final": s_final,
    }


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "investigate_results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 110)
    print("IMO-127: Investigating Hutchinson noise on 2D test functions")
    print("=" * 110)

    # -----------------------------------------------------------------------
    # Part 1: Summary table across all functions, modes, and configs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("PART 1: Convergence comparison — exact vs Hutchinson vs multi-sample")
    print("=" * 110)

    for fname, (fn, theta0, threshold) in FUNCTIONS.items():
        print(f"\n{'─' * 110}")
        print(f"  {fname}   start={np.array(theta0)}   threshold={threshold}")
        print(f"{'─' * 110}")
        header = (f"  {'mode':>12s}  {'lr':>6s}  {'clip':>5s}  {'final_loss':>12s}  "
                  f"{'steps_to_ε':>10s}  {'neg_frac':>8s}  {'clip_frac':>9s}  "
                  f"{'s_final':>20s}  {'diverged':>8s}")
        print(header)
        print(f"  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*12}  {'─'*10}  "
              f"{'─'*8}  {'─'*9}  {'─'*20}  {'─'*8}")

        for mode_name, K in MODES:
            for lr, clip_val in CONFIGS:
                hist = run_with_diagnostics(
                    fn, theta0, mode_name.split("-")[0] if "-" not in mode_name else "hutchinson",
                    clip_val, n_steps=N_STEPS, lr=lr, beta_s=0.01, mom=0.9,
                    hess_eps=1e-8, seed=42, K=K,
                )
                s = summarize(hist, threshold)
                ste_str = str(s["steps_to_eps"]) if s["steps_to_eps"] is not None else "—"
                s_str = f"[{s['s_final'][0]:+.3f}, {s['s_final'][1]:+.3f}]"
                print(f"  {mode_name:>12s}  {lr:>6.3f}  {clip_val:>5.1f}  "
                      f"{s['final_loss']:>12.3e}  {ste_str:>10s}  "
                      f"{s['neg_frac']:>8.3f}  {s['clip_frac']:>9.3f}  "
                      f"{s_str:>20s}  {'YES' if s['diverged'] else 'no':>8s}")

                # Save trajectory CSV for the best config per function
                csv_path = os.path.join(
                    out_dir,
                    f"{fname}_{mode_name}_lr{lr}_c{clip_val}.csv"
                )
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "loss", "s_0", "s_1", "s_target_0", "s_target_1",
                                "h_diag_0", "h_diag_1", "n_neg", "n_clipped",
                                "s_before_clip_0", "s_before_clip_1"])
                    for t in range(len(hist["loss"])):
                        w.writerow([
                            t,
                            hist["loss"][t],
                            hist["s"][t][0], hist["s"][t][1],
                            hist["s_target"][t][0], hist["s_target"][t][1],
                            hist["h_diag"][t][0], hist["h_diag"][t][1],
                            hist["n_neg"][t], hist["n_clipped"][t],
                            hist["s_before_clip"][t][0], hist["s_before_clip"][t][1],
                        ])

    # -----------------------------------------------------------------------
    # Part 2: Focused comparison — best config per mode for each function
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 110)
    print("PART 2: Best result per mode (best lr/clip combo)")
    print("=" * 110)

    for fname, (fn, theta0, threshold) in FUNCTIONS.items():
        print(f"\n  {fname}:")
        print(f"  {'mode':>12s}  {'best_lr':>7s}  {'best_c':>6s}  "
              f"{'final_loss':>12s}  {'steps_to_ε':>10s}  {'neg_frac':>8s}")
        print(f"  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*8}")

        for mode_name, K in MODES:
            best = None
            for lr, clip_val in CONFIGS:
                hist = run_with_diagnostics(
                    fn, theta0, mode_name.split("-")[0] if "-" not in mode_name else "hutchinson",
                    clip_val, n_steps=N_STEPS, lr=lr, beta_s=0.01, mom=0.9,
                    hess_eps=1e-8, seed=42, K=K,
                )
                s = summarize(hist, threshold)
                if best is None or (np.isfinite(s["final_loss"]) and s["final_loss"] < best[0]):
                    best = (s["final_loss"], lr, clip_val, s["steps_to_eps"], s["neg_frac"])

            fl, lr, cv, ste, nf = best
            ste_str = str(ste) if ste is not None else "—"
            print(f"  {mode_name:>12s}  {lr:>7.3f}  {cv:>6.1f}  "
                  f"{fl:>12.3e}  {ste_str:>10s}  {nf:>8.3f}")

    # -----------------------------------------------------------------------
    # Part 3: Negative estimate statistics (Hutchinson mode only)
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 110)
    print("PART 3: Hutchinson negative estimate statistics (single sample, best config)")
    print("=" * 110)

    for fname, (fn, theta0, threshold) in FUNCTIONS.items():
        # Run with c=4 (default) to see the noise
        hist = run_with_diagnostics(
            fn, theta0, "hutchinson", clip_val=4.0, n_steps=N_STEPS,
            lr=0.005, beta_s=0.01, mom=0.9, hess_eps=1e-8, seed=42, K=1,
        )
        n_steps_actual = len(hist["loss"])
        total_neg = sum(hist["n_neg"])
        neg_per_step = np.array(hist["n_neg"])
        steps_with_any_neg = np.sum(neg_per_step > 0)

        h_diag_all = np.array(hist["h_diag"])
        n_eps_clamp = np.sum(h_diag_all < 1e-8)

        s_target_all = np.array(hist["s_target"])
        s_all = np.array(hist["s"])

        print(f"\n  {fname} (lr=0.005, c=4.0, {n_steps_actual} steps):")
        print(f"    Total negative h_diag estimates:  {total_neg} / {n_steps_actual * 2} "
              f"({100*total_neg/(n_steps_actual*2):.1f}%)")
        print(f"    Steps with any negative estimate: {steps_with_any_neg} / {n_steps_actual} "
              f"({100*steps_with_any_neg/n_steps_actual:.1f}%)")
        print(f"    Eps-clamp events (h < 1e-8):      {n_eps_clamp} / {n_steps_actual * 2} "
              f"({100*n_eps_clamp/(n_steps_actual*2):.1f}%)")
        if n_steps_actual > 0:
            print(f"    s_target range: [{s_target_all.min():.2f}, {s_target_all.max():.2f}]")
            print(f"    s range (after clip): [{s_all.min():.2f}, {s_all.max():.2f}]")
            print(f"    h_diag range: [{h_diag_all.min():.4e}, {h_diag_all.max():.4e}]")
            print(f"    Final loss: {hist['loss'][-1]:.6e}")

    print(f"\n{'=' * 110}")
    print(f"Trajectory CSVs written to: {out_dir}/")
    print("Done.")
