"""
IMO-87: Numerical verification — optimal embedding with Newton-targeted metric.

Compares three embedding choices (plain f=L, log f=log L, identity xi=0)
on diagonal quadratics with Newton-targeted diagonal scaling s_i.

Produces:
  ../images/embedding_convergence.png   — loss curves
  ../images/embedding_damping.png       — scalar damping r_t
  ../images/embedding_eta_sweep.png     — convergence vs eta for each embedding
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "images"
IMG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate(h, theta0, embedding, eta, xi, mu, lam, clip, n_steps):
    """Run the induced-metric optimizer on L = (1/2) theta^T diag(h) theta.

    Args:
        h: diagonal Hessian eigenvalues (1D array)
        theta0: initial parameters
        embedding: "plain", "log", or "identity"
        eta: learning rate
        xi: metric strength
        mu: metric learning rate
        lam: metric regularization
        clip: s clipping bound
        n_steps: number of steps

    Returns:
        loss_hist, r_hist, s_hist (arrays)
    """
    n = len(h)
    theta = theta0.copy()
    s = np.zeros(n)
    eps = 1e-8

    loss_hist = np.zeros(n_steps)
    r_hist = np.zeros(n_steps)
    s_hist = np.zeros((n_steps, n))

    # Newton-targeted driving force (constant for quadratic)
    drive = -lam * np.log(np.maximum(h, eps))

    for t in range(n_steps):
        g = h * theta
        loss = 0.5 * np.sum(h * theta**2)
        v = xi * np.sum(np.exp(s) * g**2)

        if embedding == "plain":
            r = 1.0 / (1.0 + abs(v))
        elif embedding == "log":
            L = max(loss, 1e-300)
            r = L / (L**2 + abs(v) + 1e-300)
        elif embedding == "identity":
            r = 1.0
        else:
            raise ValueError(f"Unknown embedding: {embedding}")

        # Parameter update
        theta = theta - eta * r * np.exp(s) * g

        # Metric update (Newton-targeted)
        s = (1 - mu * lam) * s + mu * drive
        s = s - np.mean(s)  # mean-center
        s = np.clip(s, -clip, clip)

        loss_hist[t] = 0.5 * np.sum(h * theta**2)
        r_hist[t] = r
        s_hist[t] = s.copy()

    return loss_hist, r_hist, s_hist


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

N = 10
h = np.logspace(0, 2, N)  # eigenvalues from 1 to 100, kappa = 100
h_geo = np.exp(np.mean(np.log(h)))
theta0 = np.ones(N)

xi = 0.1
mu = 0.01
lam = 0.01
clip = 6.0

print(f"N = {N}, kappa = {h[-1]/h[0]:.0f}, h_geo = {h_geo:.2f}")
print(f"Newton equilibrium: all e^s_i * h_i = {h_geo:.2f}")
print(f"Metric timescale: 1/(mu*lam) = {1/(mu*lam):.0f} steps")
print()


# ---------------------------------------------------------------------------
# Experiment 1: Convergence comparison at fixed eta
# ---------------------------------------------------------------------------

eta = 0.005  # stable for all three
n_steps = 15000

embeddings = ["plain", "log", "identity"]
colors = {"plain": "C0", "log": "C1", "identity": "C2"}
labels = {"plain": r"Plain ($f = L$)", "log": r"Log ($f = \log L$)",
          "identity": r"Identity ($\xi = 0$)"}

results = {}
for emb in embeddings:
    loss, r, s = simulate(h, theta0, emb, eta, xi, mu, lam, clip, n_steps)
    results[emb] = {"loss": loss, "r": r, "s": s}
    final = loss[-1] if loss[-1] > 0 else float("inf")
    print(f"{emb:10s}: final loss = {final:.3e}, final r = {r[-1]:.6f}")

# --- Figure 1: Convergence curves ---
fig, ax = plt.subplots(figsize=(8, 5))
for emb in embeddings:
    loss = results[emb]["loss"]
    # Replace zeros for log plot
    loss_plot = np.where(loss > 0, loss, np.nan)
    ax.semilogy(loss_plot, color=colors[emb], label=labels[emb], linewidth=1.5)

ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title(rf"Embedding comparison on {N}D quadratic ($\kappa = {h[-1]/h[0]:.0f}$, "
             rf"$\eta = {eta}$, $\xi = {xi}$)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_DIR / "embedding_convergence.png", dpi=150)
print(f"\nSaved {IMG_DIR / 'embedding_convergence.png'}")

# --- Figure 2: Damping factor ---
fig, ax = plt.subplots(figsize=(8, 5))
for emb in embeddings:
    r = results[emb]["r"]
    ax.semilogy(r, color=colors[emb], label=labels[emb], linewidth=1.5)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label=r"$r = 1$")
ax.axhline(1.0 / (2 * xi * h_geo), color=colors["log"], linestyle=":",
           linewidth=0.8, label=rf"$1/(2\xi h_{{geo}}) = {1/(2*xi*h_geo):.1f}$")
ax.set_xlabel("Step")
ax.set_ylabel(r"Damping factor $r_t$")
ax.set_title("Scalar damping factor evolution")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_DIR / "embedding_damping.png", dpi=150)
print(f"Saved {IMG_DIR / 'embedding_damping.png'}")


# ---------------------------------------------------------------------------
# Experiment 2: Stability — max eta before divergence
# ---------------------------------------------------------------------------

print("\n--- Stability analysis: max stable eta ---")

def find_max_stable_eta(embedding, h, theta0, xi, mu, lam, clip,
                        n_steps=5000, tol=1e10):
    """Binary search for the max eta that doesn't diverge."""
    lo, hi = 1e-6, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        loss, _, _ = simulate(h, theta0, embedding, mid, xi, mu, lam, clip, n_steps)
        if np.isfinite(loss[-1]) and loss[-1] < tol:
            lo = mid
        else:
            hi = mid
    return lo

for emb in embeddings:
    eta_max = find_max_stable_eta(emb, h, theta0, xi, mu, lam, clip)
    print(f"{emb:10s}: max stable eta ~ {eta_max:.4f}")

print(f"Theoretical identity limit: 2/max(h) = {2/h[-1]:.4f}")
print(f"Plain can use larger eta due to geometric clipping.")


# ---------------------------------------------------------------------------
# Experiment 3: Eta sweep — convergence rate vs learning rate
# ---------------------------------------------------------------------------

print("\n--- Eta sweep ---")
etas = np.logspace(-3, -0.5, 30)
n_steps_sweep = 10000

fig, ax = plt.subplots(figsize=(8, 5))

for emb in embeddings:
    final_losses = []
    for et in etas:
        try:
            loss, _, _ = simulate(h, theta0, emb, et, xi, mu, lam, clip, n_steps_sweep)
            fl = loss[-1] if np.isfinite(loss[-1]) else 1e20
        except Exception:
            fl = 1e20
        final_losses.append(fl)

    final_losses = np.array(final_losses)
    # Replace very large / nan values
    final_losses = np.where(final_losses < 1e10, final_losses, np.nan)
    final_losses = np.where(final_losses > 0, final_losses, 1e-320)
    ax.semilogy(etas, final_losses, "o-", color=colors[emb], label=labels[emb],
                markersize=3, linewidth=1.5)

ax.set_xlabel(r"Learning rate $\eta$")
ax.set_ylabel(f"Loss at step {n_steps_sweep}")
ax.set_title(rf"Convergence vs $\eta$ ({N}D quadratic, $\kappa = {h[-1]/h[0]:.0f}$)")
ax.set_xscale("log")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_DIR / "embedding_eta_sweep.png", dpi=150)
print(f"Saved {IMG_DIR / 'embedding_eta_sweep.png'}")


# ---------------------------------------------------------------------------
# Experiment 4: Asymptotic convergence rate (after metric converges)
# ---------------------------------------------------------------------------

print("\n--- Asymptotic convergence rate ---")
print("At Newton equilibrium, contraction = |1 - eta * r_inf * h_geo|")
print(f"h_geo = {h_geo:.2f}")

for emb in ["plain", "identity"]:
    r_inf = 1.0
    eta_opt = 1.0 / h_geo
    print(f"\n{emb}: r_inf = {r_inf}, eta_opt = 1/h_geo = {eta_opt:.4f}")
    print(f"  contraction at eta={eta}: rho = |1 - {eta} * {r_inf} * {h_geo:.1f}| "
          f"= {abs(1 - eta * r_inf * h_geo):.4f}")

r_inf_log = 1.0 / (2 * xi * h_geo)
eta_opt_log = 2 * xi
print(f"\nlog: r_inf = 1/(2*xi*h_geo) = {r_inf_log:.4f}, "
      f"eta_opt = 2*xi = {eta_opt_log:.4f}")
print(f"  contraction at eta={eta}: rho = |1 - {eta} * {r_inf_log:.4f} * {h_geo:.1f}| "
      f"= {abs(1 - eta * r_inf_log * h_geo):.4f}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
1. Plain (f=L): r -> 1 as L -> 0. Fully recovers Newton step.
   Self-stabilising (geometric clipping). RECOMMENDED.

2. Log (f=log L): r -> 1/(2*xi*h_geo) as L -> 0.
   Permanently caps effective step size. Optimal eta depends
   on xi, not curvature. Newton equalization is invisible.

3. Identity (xi=0): r = 1 always. Pure Newton, no interference.
   But no safety net — max stable eta = 2/max(h_i), very
   restrictive for ill-conditioned problems during transient.

Plain embedding is the unique choice that:
  - Doesn't interfere with Newton preconditioning (r -> 1)
  - Provides geometric clipping as safety net (r << 1 early)
""")
