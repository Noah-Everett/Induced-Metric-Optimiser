"""
Clip sweep on classic test functions (JIT-compiled inner loop).

Tests how metric_clip (c) affects the derived Newton-targeted optimizer
on Rosenbrock, Beale, Himmelblau, Ackley, and the symmetric saddle.

Run: PYTHONPATH=repo .conda/bin/python repo/notebooks/clip_sweep_test_functions.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

jax.config.update("jax_enable_x64", True)

from optimisers.jax_derived_newton_diag import derived_newton_diag
from optimisers.jax_diag_utils import apply_diag, mean_center, clip as clip_s


# ---------------------------------------------------------------------------
# Test functions
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


# ---------------------------------------------------------------------------
# JIT-compiled training loop
# ---------------------------------------------------------------------------

def make_train_step(loss_fn, lr, mom_coeff, beta_s, clip_val, hess_eps):
    """Build a JIT-compiled single training step."""
    neg_lr = -lr
    one_minus_mom = 1.0 - mom_coeff
    one_minus_bs = 1.0 - beta_s
    lo, hi = -abs(clip_val), abs(clip_val)

    @jax.jit
    def step(theta, s, m, step_count, key):
        step_count = step_count + 1

        # Gradient
        g = jax.grad(loss_fn)(theta)

        # Hutchinson HVP diagonal
        key, subkey = jax.random.split(key)
        v = 2.0 * jax.random.bernoulli(subkey, 0.5, theta.shape).astype(theta.dtype) - 1.0
        _, hvp = jax.jvp(jax.grad(loss_fn), (theta,), (v,))
        h_diag = v * hvp

        # Metric EMA
        s_new = one_minus_bs * s + beta_s * (-jnp.log(jnp.maximum(h_diag, hess_eps)))
        s_new = s_new - jnp.mean(s_new)
        s_new = jnp.clip(s_new, lo, hi)

        # Momentum + bias correction
        m_new = mom_coeff * m + one_minus_mom * g
        m_corr = m_new / (1.0 - mom_coeff ** step_count)

        # Preconditioned step
        theta_new = theta + neg_lr * jnp.exp(s_new) * m_corr

        loss = loss_fn(theta_new)
        return theta_new, s_new, m_new, step_count, key, loss

    return step


def run_fast(loss_fn, theta0, clip_val, n_steps=2000, lr=0.01, beta_s=0.01,
             mom=0.9, hess_eps=1e-8, seed=42):
    """Run optimizer with JIT-compiled step. Returns (final_loss, s_final, diverged)."""
    step_fn = make_train_step(loss_fn, lr, mom, beta_s, clip_val, hess_eps)

    theta = theta0.copy()
    s = jnp.zeros_like(theta)
    m = jnp.zeros_like(theta)
    sc = jnp.array(0, dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)

    # Warmup JIT
    theta_w, s_w, m_w, sc_w, key_w, _ = step_fn(theta, s, m, sc, key)

    # Reset and run
    theta = theta0.copy()
    s = jnp.zeros_like(theta)
    m = jnp.zeros_like(theta)
    sc = jnp.array(0, dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)

    losses = []
    for t in range(n_steps):
        theta, s, m, sc, key, loss = step_fn(theta, s, m, sc, key)
        lv = float(loss)
        losses.append(lv)
        if not np.isfinite(lv) or lv > 1e20:
            return lv, np.array(s), True, losses

    return float(losses[-1]), np.array(s), False, losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FUNCTIONS = {
    "Rosenbrock": (rosenbrock, jnp.array([-1.2, 1.0]),  1e-6),
    "Beale":      (beale,      jnp.array([0.5, 0.5]),    1e-6),
    "Himmelblau": (himmelblau, jnp.array([0.0, 0.0]),    1e-6),
    "Ackley":     (ackley,     jnp.array([0.5, 0.5]),    1e-4),
    "Saddle":     (saddle,     jnp.array([1.0, 1.0]),    None),
}

clip_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
lrs = [0.001, 0.005, 0.01, 0.05]

if __name__ == "__main__":
    print("=" * 85)
    print("Clip sweep on classic test functions (JIT-compiled)")
    print("=" * 85)

    for fname, (fn, theta0, threshold) in FUNCTIONS.items():
        print(f"\n{'─' * 85}")
        print(f"  {fname}   start={np.array(theta0)}   threshold={threshold}")
        print(f"{'─' * 85}")
        print(f"  {'clip':>5s}  {'lr':>6s}  {'final_loss':>12s}  "
              f"{'steps_to_ε':>10s}  {'s_min':>8s}  {'s_max':>8s}  "
              f"{'clipped?':>8s}  {'diverged?':>9s}")
        print(f"  {'─'*5}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*8}  "
              f"{'─'*8}  {'─'*8}  {'─'*9}")

        for clip_val in clip_values:
            best_loss = float('inf')
            best_result = None

            for lr in lrs:
                t0 = time.time()
                final_loss, s_final, diverged, losses = run_fast(
                    fn, theta0, clip_val, n_steps=2000, lr=lr,
                )
                dt = time.time() - t0

                if np.isfinite(final_loss) and final_loss < best_loss:
                    steps_to_eps = None
                    if threshold is not None and not diverged:
                        for i, l in enumerate(losses):
                            if l < threshold:
                                steps_to_eps = i
                                break

                    s_min = float(s_final.min())
                    s_max = float(s_final.max())
                    at_clip = (abs(abs(s_max) - clip_val) < 0.1 or
                               abs(abs(s_min) - clip_val) < 0.1)

                    best_loss = final_loss
                    best_result = (clip_val, lr, final_loss, steps_to_eps,
                                   s_min, s_max, at_clip, diverged)

            if best_result is not None:
                cv, lr, fl, ste, smin, smax, at_c, div = best_result
                ste_str = str(ste) if ste is not None else "—"
                clip_str = "YES" if at_c else "no"
                div_str = "YES" if div else "no"
                print(f"  {cv:>5.0f}  {lr:>6.3f}  {fl:>12.3e}  "
                      f"{ste_str:>10s}  {smin:>8.3f}  {smax:>8.3f}  "
                      f"{clip_str:>8s}  {div_str:>9s}")
            else:
                print(f"  {clip_val:>5.0f}  ALL DIVERGED")

    print(f"\n{'=' * 85}")
    print("Done.")
