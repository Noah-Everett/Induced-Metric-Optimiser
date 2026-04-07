"""
IMO-48: Numerical + analytical verification of theoretical HP constraints.

Analytical results (Mathematica-verified):
  - Equilibrium: s* = -W(-xi*g^2*(1 - r*tanh(H/tau))/lam)  [Lambert W]
  - Sign-flip condition: r_crit = coth(H/tau), approaches 1 for |H| >> tau
  - Mean-centered bifurcation slightly below coth(H/tau)
  - tau sensitivity: d/dtau[2 tanh(H/tau)] = -2H sech^2(H/tau)/tau^2,
    maximized at tau ~ |H|

Numerical verification of four constraint claims:
  1. Sign-flip transition at curv_ratio ~ coth(H/tau)
  2. metric_clip: when does it bind under mean-centering?
  3. curv_tau useful range ~ [0.5, 10]
  4. beta < 0.8 gives noisy denominator EMA
"""
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 11})

OUT = "notebooks/hp-constraint-verification.png"


# ================================================================
# Helpers
# ================================================================

def equilibrium_s_single(ratio, H, xi=0.1, g2=0.1, tau=1.0, lam=1e-3, clip=4.0):
    """Single-component equilibrium (no mean-centering)."""
    coeff = xi * g2 * (1 - ratio * np.tanh(H / tau))
    def f(s):
        return coeff * np.exp(s) - lam * s
    lo, hi = -clip, clip
    fa, fb = f(lo), f(hi)
    if fa * fb > 0:
        return hi if coeff > 0 else lo
    return brentq(f, lo, hi)


def simulate_metric_deterministic(
    N, n_steps, xi, mu, lam, clip, curv_beta, tau, H_diag, g2_diag,
):
    """Deterministic metric dynamics with mean-centering (no gradient noise)."""
    s = np.zeros(N)
    for _ in range(n_steps):
        curv_sign = np.tanh(H_diag / tau)
        s_grad = xi * np.exp(s) * g2_diag - curv_beta * curv_sign * np.exp(s) * g2_diag
        s = s + mu * s_grad - mu * lam * s
        s = s - np.mean(s)
        s = np.clip(s, -clip, clip)
    return s


# ================================================================
# Plots
# ================================================================

ratios = np.linspace(0.02, 5.0, 300)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ---- Panel 1: Sign-flip transition ----
ax = axes[0, 0]
for H, ls, color, label in [
    (5.0, "-", "tab:blue", "H=+5 (convex)"),
    (1.0, "--", "tab:blue", "H=+1 (mod convex)"),
    (-1.0, "--", "tab:red", r"H=$-$1 (mod concave)"),
    (-5.0, "-", "tab:red", r"H=$-$5 (concave)"),
]:
    ss = [equilibrium_s_single(r, H) for r in ratios]
    ax.plot(ratios, ss, ls=ls, color=color, label=label)

# Analytical r_crit = coth(H/tau)
for H_val, ypos in [(5.0, -3.5), (1.0, -2.0)]:
    rc = 1.0 / np.tanh(H_val / 1.0)
    ax.axvline(rc, color="blue", ls=":", lw=0.8, alpha=0.5)
    ax.annotate(f"coth({H_val:.0f})={rc:.3f}", xy=(rc + 0.05, ypos), fontsize=8, color="blue")

ax.axhline(0, color="gray", lw=0.5)
ax.axvspan(0.2, 5.0, alpha=0.08, color="green")
ax.set_xlabel(r"curv_ratio  ($\beta_c / \xi$)")
ax.set_ylabel("equilibrium s*")
ax.set_title(r"1. Sign-flip: $s^*=0$ at $r = \coth(H/\tau)$")
ax.legend(fontsize=9)
ax.set_xlim(0, 5)


# ---- Panel 2: metric_clip — deterministic multi-dim ----
ax = axes[0, 1]
N = 20
H_diag = np.linspace(-5, 5, N)
g2_diag = np.ones(N) * 0.1

# Sweep curv_ratio to show when clip binds
curv_ratios_test = [0.5, 1.0, 1.5, 2.0, 3.0]
colors_r = plt.cm.viridis(np.linspace(0.15, 0.85, len(curv_ratios_test)))

for cr, color in zip(curv_ratios_test, colors_r):
    xi_val = 0.1
    cb = cr * xi_val
    s_final = simulate_metric_deterministic(
        N=N, n_steps=20000, xi=xi_val, mu=1e-3, lam=1e-3,
        clip=4.0, curv_beta=cb, tau=1.0, H_diag=H_diag, g2_diag=g2_diag,
    )
    ax.plot(H_diag, s_final, "o-", ms=3, color=color, label=f"r={cr}")

ax.axhline(0, color="gray", lw=0.5)
ax.axhline(4.0, color="red", ls=":", lw=0.8, alpha=0.4)
ax.axhline(-4.0, color="red", ls=":", lw=0.8, alpha=0.4)
ax.set_xlabel("Hessian diagonal $H_{ii}$")
ax.set_ylabel("equilibrium $s_i$ (mean-centered)")
ax.set_title("2. Metric profile vs curv_ratio (clip=4, 20D)")
ax.legend(fontsize=9)


# ---- Panel 3: curv_tau useful range ----
ax = axes[1, 0]
taus = np.logspace(-1, 2, 200)
H_vals = [0.5, 1.0, 5.0, 10.0]

for H in H_vals:
    diff = 2 * np.tanh(H / taus)
    ax.semilogx(taus, diff, label=f"|H|={H}")

ax.axvspan(0.5, 10.0, alpha=0.15, color="green", label="proposed range [0.5, 10]")
ax.axvspan(0.1, 0.5, alpha=0.1, color="red")
ax.axvspan(10.0, 100.0, alpha=0.1, color="red")
ax.set_xlabel(r"curv_tau ($\tau$)")
ax.set_ylabel(r"sign discrimination: $\tanh(H/\tau) - \tanh(-H/\tau)$")
ax.set_title(r"3. $\tau$ useful range $\approx$ [0.5, 10]")
ax.legend(fontsize=9)
ax.annotate("saturated\n(noisy)", xy=(0.15, 0.3), fontsize=8, color="red")
ax.annotate("negligible\ncorrection", xy=(30, 0.3), fontsize=8, color="red")


# ---- Panel 4: beta noise ----
ax = axes[1, 1]
np.random.seed(42)
n_steps = 5000
xi_val = 0.1

betas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
noise_levels = []

for beta_val in betas:
    metric_ema = 0.0
    v_hats = []
    for t in range(1, n_steps + 1):
        g = np.random.randn() * 2 + 1
        v = xi_val * g ** 2
        metric_ema = beta_val * metric_ema + (1 - beta_val) * v
        v_hat = metric_ema / (1 - beta_val ** t)
        v_hats.append(v_hat)
    v_arr = np.array(v_hats[500:])
    cv = np.std(v_arr) / np.mean(v_arr)
    noise_levels.append(cv)

bars = ax.bar(
    [str(b) for b in betas], noise_levels,
    color=["red" if b < 0.8 else "tab:blue" for b in betas],
)
ax.set_xlabel("beta (EMA coefficient)")
ax.set_ylabel(r"coefficient of variation of $\hat{v}$")
ax.set_title(r"4. Denominator noise vs $\beta$")
ax.axhline(noise_levels[betas.index(0.8)], color="green", ls="--", lw=1,
           label=r"$\beta$=0.8 threshold")
ax.legend()

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved: {OUT}")


# ================================================================
# Numerical summary
# ================================================================

print("\n" + "=" * 70)
print("KEY NUMERICAL RESULTS")
print("=" * 70)

print("\n1. SIGN-FLIP TRANSITION")
print("   Analytical: r_crit = coth(H/tau)")
for H_val in [1.0, 2.0, 5.0, 10.0]:
    rc = 1.0 / np.tanh(H_val / 1.0)
    print(f"   H={H_val:4.0f}, tau=1: r_crit = {rc:.6f}")
print("\n   Single-component equilibria:")
for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    s_conv = equilibrium_s_single(r, 5.0)
    s_conc = equilibrium_s_single(r, -5.0)
    print(f"   r={r:.1f}: s*(H=+5)={s_conv:+.3f}, s*(H=-5)={s_conc:+.3f}")

print("\n2. METRIC_CLIP BINDING (deterministic, 20D, mean-centered)")
for cr in [0.5, 1.0, 1.5, 2.0, 3.0]:
    xi_val = 0.1
    s_final = simulate_metric_deterministic(
        N=20, n_steps=20000, xi=xi_val, mu=1e-3, lam=1e-3,
        clip=4.0, curv_beta=cr * xi_val, tau=1.0,
        H_diag=np.linspace(-5, 5, 20), g2_diag=np.ones(20) * 0.1,
    )
    max_dev = np.max(np.abs(s_final))
    at_clip = np.sum(np.abs(np.abs(s_final) - 4.0) < 0.05)
    print(f"   r={cr:.1f}: max|s|={max_dev:.3f}, at_clip={at_clip}/20, "
          f"metric_ratio=exp(2*{max_dev:.2f})={np.exp(2 * max_dev):.1f}")

# Test clip sensitivity at high curv_ratio
print("\n   Clip sensitivity at r=2.0:")
for clip_val in [1.0, 2.0, 3.0, 4.0, 5.0]:
    s_final = simulate_metric_deterministic(
        N=20, n_steps=20000, xi=0.1, mu=1e-3, lam=1e-3,
        clip=clip_val, curv_beta=0.2, tau=1.0,
        H_diag=np.linspace(-5, 5, 20), g2_diag=np.ones(20) * 0.1,
    )
    max_dev = np.max(np.abs(s_final))
    at_clip = np.sum(np.abs(np.abs(s_final) - clip_val) < 0.05)
    print(f"   clip={clip_val:.0f}: max|s|={max_dev:.3f}, at_clip={at_clip}/20")

print("\n3. CURV_TAU DISCRIMINATION (|H|=1):")
for tau in [0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
    disc = 2 * np.tanh(1.0 / tau)
    sens = 2 * 1.0 / (np.cosh(1.0 / tau) ** 2 * tau ** 2)  # |d/dtau|
    print(f"   tau={tau:6.1f}: discrimination={disc:.4f}, sensitivity={sens:.4f}")

print("\n4. BETA NOISE (CV of v_hat):")
for b, cv in zip(betas, noise_levels):
    print(f"   beta={b:.2f}: CV={cv:.3f}")
