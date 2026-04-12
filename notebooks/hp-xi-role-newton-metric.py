#!/usr/bin/env python3
"""
IMO-86: Numerical verification — role of ξ with Newton-targeted metric.

Runs the full optimizer on N-dimensional quadratics with Newton-targeted s_i,
varying ξ to verify the analytical predictions:
  - Plain embedding: ξ=0 is optimal, ξ>0 slows convergence
  - Log-loss embedding: ξ=0 is singular, optimal η = 2ξ
  - Newton metric equalizes all contraction factors
"""

import numpy as np
from dataclasses import dataclass


# ── Simplified optimizer (captures the essential dynamics) ────────────────────

@dataclass
class State:
    theta: np.ndarray
    momentum: np.ndarray
    metric_ema: float
    log_diag: np.ndarray  # s_i values
    step: int


def newton_target(h, eps=1e-8):
    """Compute Newton-targeted s_i = -log(max(h_i, eps)) + mean(log(max(h, eps)))."""
    log_h = np.log(np.maximum(h, eps))
    return -log_h + np.mean(log_h)


def run_optimizer(h, theta0, xi, eta, beta_m=0.9, beta=0.8,
                  n_steps=200, log_loss=False, use_newton_s=True,
                  metric_lr=0.0, metric_reg=0.0):
    """
    Run induced-metric optimizer on quadratic L = 0.5 * theta^T H theta.

    Parameters
    ----------
    h : array, Hessian diagonal
    theta0 : array, initial parameters
    xi : float, metric strength
    eta : float, learning rate
    log_loss : bool, use log-loss embedding
    use_newton_s : bool, use Newton-targeted s_i (vs s_i=0)
    """
    N = len(h)
    if use_newton_s:
        s = newton_target(h)
    else:
        s = np.zeros(N)

    theta = theta0.copy()
    mom = np.zeros(N)
    v_ema = 0.0
    losses = []

    for t in range(1, n_steps + 1):
        # Loss and gradient
        loss = 0.5 * np.sum(h * theta**2)
        g = h * theta
        losses.append(loss)

        # Trace statistic
        alpha = np.exp(s)
        g_tilde = alpha * g
        v = xi * np.sum(g * g_tilde)

        # Denominator EMA
        v_ema = beta * v_ema + (1 - beta) * v
        v_hat = v_ema / (1 - beta**t)

        # Damping factor
        if log_loss:
            loss_val = max(loss, 1e-30)
            r = loss_val / (loss_val**2 + abs(v_hat))
        else:
            r = 1.0 / (1.0 + abs(v_hat))

        # Momentum
        mom = beta_m * mom + (1 - beta_m) * g
        m_corr = mom / (1 - beta_m**t)
        m_tilde = alpha * m_corr

        # Update
        theta = theta - eta * r * m_tilde

        # Optionally adapt s (gradient-magnitude drive for comparison)
        if metric_lr > 0 and not use_newton_s:
            s_grad = xi * alpha * g**2
            s = s + metric_lr * s_grad - metric_lr * metric_reg * s
            s = s - np.mean(s)  # mean-center
            s = np.clip(s, -4, 4)

    losses.append(0.5 * np.sum(h * theta**2))
    return np.array(losses)


# ── Experiment 1: Plain embedding, vary ξ ─────────────────────────────────────

def experiment_plain_vary_xi():
    print("=" * 70)
    print("EXPERIMENT 1: Plain embedding, Newton-targeted s_i, varying ξ")
    print("=" * 70)
    N = 20
    np.random.seed(42)
    h = np.exp(np.linspace(np.log(0.1), np.log(100), N))  # κ = 1000
    theta0 = np.random.randn(N)
    h_geo = np.exp(np.mean(np.log(h)))
    eta = 0.8 / h_geo  # slightly less than 1/h_geo

    print(f"N={N}, κ={h.max()/h.min():.0f}, h_geo={h_geo:.4f}, η={eta:.4f}")
    print(f"L₀ = {0.5*np.sum(h*theta0**2):.4f}")
    print()

    xi_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    for xi in xi_values:
        losses = run_optimizer(h, theta0, xi=xi, eta=eta, n_steps=100)
        print(f"  ξ={xi:>6.3f}: L_50={losses[50]:.2e}, L_100={losses[100]:.2e}")

    print()
    print("PREDICTION: ξ=0 should converge fastest.")
    print()


# ── Experiment 2: Log-loss embedding, vary ξ ──────────────────────────────────

def experiment_log_loss_vary_xi():
    print("=" * 70)
    print("EXPERIMENT 2: Log-loss embedding, Newton-targeted s_i, varying ξ")
    print("=" * 70)
    N = 20
    np.random.seed(42)
    h = np.exp(np.linspace(np.log(0.1), np.log(100), N))
    theta0 = np.random.randn(N)
    h_geo = np.exp(np.mean(np.log(h)))

    print(f"N={N}, κ={h.max()/h.min():.0f}, h_geo={h_geo:.4f}")
    print()

    # Vary ξ with η = 2ξ (optimal pairing)
    print("--- η = 2ξ (optimal pairing) ---")
    xi_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    for xi in xi_values:
        eta = 2 * xi
        losses = run_optimizer(h, theta0, xi=xi, eta=eta, n_steps=200, log_loss=True)
        print(f"  ξ={xi:>5.2f}, η={eta:>5.2f}: L_100={losses[100]:.2e}, L_200={losses[200]:.2e}")

    print()

    # Fixed η, vary ξ
    print("--- Fixed η=1.0, varying ξ ---")
    eta = 1.0
    xi_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    for xi in xi_values:
        losses = run_optimizer(h, theta0, xi=xi, eta=eta, n_steps=200, log_loss=True)
        rate = (1 - eta / (2 * xi))**2
        print(f"  ξ={xi:>5.2f}: L_200={losses[200]:.2e}, predicted rate=(1-η/(2ξ))²={rate:.4f}")

    print()
    print("PREDICTION: Optimal ξ = η/2 = 0.5 (gives zero asymptotic rate).")
    print()


# ── Experiment 3: Newton-targeted vs gradient-magnitude, ξ interaction ────────

def experiment_newton_vs_gradient():
    print("=" * 70)
    print("EXPERIMENT 3: Newton-targeted vs gradient-magnitude s_i")
    print("=" * 70)
    N = 20
    np.random.seed(42)
    h = np.exp(np.linspace(np.log(0.1), np.log(100), N))
    theta0 = np.random.randn(N)
    h_geo = np.exp(np.mean(np.log(h)))
    eta = 0.8 / h_geo

    print(f"N={N}, κ={h.max()/h.min():.0f}, h_geo={h_geo:.4f}, η={eta:.4f}")
    print()

    # Newton-targeted, ξ=0
    losses_newton_0 = run_optimizer(h, theta0, xi=0.0, eta=eta, n_steps=200,
                                    use_newton_s=True)
    print(f"  Newton s_i, ξ=0:   L_100={losses_newton_0[100]:.2e}, L_200={losses_newton_0[200]:.2e}")

    # Newton-targeted, ξ=0.1
    losses_newton_xi = run_optimizer(h, theta0, xi=0.1, eta=eta, n_steps=200,
                                     use_newton_s=True)
    print(f"  Newton s_i, ξ=0.1: L_100={losses_newton_xi[100]:.2e}, L_200={losses_newton_xi[200]:.2e}")

    # No metric (s=0), ξ=0.1
    losses_identity = run_optimizer(h, theta0, xi=0.1, eta=eta, n_steps=200,
                                    use_newton_s=False)
    print(f"  s_i=0,      ξ=0.1: L_100={losses_identity[100]:.2e}, L_200={losses_identity[200]:.2e}")

    # Gradient-magnitude s_i, ξ=0.1
    losses_grad = run_optimizer(h, theta0, xi=0.1, eta=eta, n_steps=200,
                                use_newton_s=False, metric_lr=1e-3, metric_reg=1e-4)
    print(f"  Grad-mag s_i, ξ=0.1: L_100={losses_grad[100]:.2e}, L_200={losses_grad[200]:.2e}")

    print()
    print("PREDICTION: Newton ξ=0 >> Newton ξ>0 >> identity >> grad-magnitude.")
    print()


# ── Experiment 4: Verify effective learning rate for log-loss ─────────────────

def experiment_effective_lr():
    print("=" * 70)
    print("EXPERIMENT 4: Log-loss effective learning rate η_eff = η/(2ξ)")
    print("=" * 70)
    N = 10
    np.random.seed(42)
    h = np.exp(np.linspace(np.log(1), np.log(10), N))
    theta0 = 0.01 * np.ones(N)  # start VERY close to minimum
    h_geo = np.exp(np.mean(np.log(h)))

    print(f"Starting near minimum: L₀ = {0.5*np.sum(h*theta0**2):.6f}")
    print(f"h_geo = {h_geo:.4f}")
    print()

    # For different (η, ξ) pairs with same η/(2ξ) = 0.5
    pairs = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
    print("--- Same η/(2ξ) = 0.5 —  should give same asymptotic rate ---")
    for eta, xi in pairs:
        losses = run_optimizer(h, theta0, xi=xi, eta=eta, n_steps=50, log_loss=True)
        # Measure empirical rate from last few steps
        if losses[-1] > 0 and losses[-5] > 0:
            rate = (losses[-1] / losses[-5])**(1/4)
        else:
            rate = 0.0
        predicted = (1 - eta / (2 * xi))**2
        print(f"  η={eta:.1f}, ξ={xi:.1f}: L_50={losses[50]:.2e}, "
              f"empirical_rate={rate:.4f}, predicted={predicted:.4f}")

    print()
    # Optimal: η/(2ξ) = 1
    pairs_opt = [(1.0, 0.5), (2.0, 1.0), (4.0, 2.0)]
    print("--- Optimal η = 2ξ → one-step convergence near minimum ---")
    for eta, xi in pairs_opt:
        losses = run_optimizer(h, theta0, xi=xi, eta=eta, n_steps=20, log_loss=True)
        print(f"  η={eta:.1f}, ξ={xi:.1f}: L_5={losses[5]:.2e}, L_20={losses[20]:.2e}")
    print()


# ── Experiment 5: Safety (large η) ───────────────────────────────────────────

def experiment_safety():
    print("=" * 70)
    print("EXPERIMENT 5: Safety — ξ prevents divergence for large η")
    print("=" * 70)
    N = 10
    np.random.seed(42)
    h = np.exp(np.linspace(np.log(1), np.log(10), N))
    theta0 = np.random.randn(N)
    h_geo = np.exp(np.mean(np.log(h)))
    eta = 2.5 / h_geo  # η h_geo = 2.5 > 2 (unstable for ξ=0)

    print(f"η h_geo = {eta * h_geo:.1f} > 2 (unstable for ξ=0)")
    print()

    for xi in [0.0, 0.01, 0.1, 1.0]:
        losses = run_optimizer(h, theta0, xi=xi, eta=eta, n_steps=100)
        final = losses[-1]
        status = "DIVERGED" if final > losses[0] else f"L_100={final:.2e}"
        print(f"  ξ={xi:>5.2f}: {status}")

    print()
    print("ξ>0 saves you when η is too large, but with Newton s_i")
    print("you know h_geo and can simply set η < 2/h_geo.")
    print()


if __name__ == "__main__":
    experiment_plain_vary_xi()
    experiment_log_loss_vary_xi()
    experiment_newton_vs_gradient()
    experiment_effective_lr()
    experiment_safety()

    print("=" * 70)
    print("OVERALL CONCLUSIONS")
    print("=" * 70)
    print("""
Plain embedding + Newton-targeted s_i:
  - ξ=0 is strictly optimal: no damping, pure Newton preconditioning
  - ξ>0 slows convergence by damping the Newton step
  - Safety benefit (preventing divergence for large η) is redundant
    because Newton s_i provides h_geo, allowing correct η selection

Log-loss embedding + Newton-targeted s_i:
  - ξ=0 is SINGULAR (r diverges as L→0)
  - ξ must be nonzero; optimal pairing: η = 2ξ
  - Effective near-minimum LR = η/(2ξ), independent of h_geo
  - The Newton metric still equalizes contraction factors,
    but ξ sets the overall convergence speed

RECOMMENDATION:
  Plain embedding:  ELIMINATE ξ (set ξ=0, simplify to Newton + momentum + η)
  Log-loss embedding: KEEP ξ, set η = 2ξ (or tune η/ξ ratio)
""")
