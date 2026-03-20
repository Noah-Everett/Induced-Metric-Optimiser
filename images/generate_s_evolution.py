"""
Generate plots showing how the learnable metric parameter s evolves during training.

- s_evolution_scalar.png : scalar s vs training step  (learnable scalar optimiser)
- s_evolution_diag.png   : several s_i vs training step (learnable diagonal optimiser)

Uses a tiny 2-layer MLP on synthetic regression so the diagonal variant has
enough distinct parameters to make the per-component s_i plot interesting.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from optimisers.learnable_scalar import custom_sgd_learnable_scalar
from optimisers.learnable_diag import custom_sgd_learnable_diag

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Plot style (matches generate_step_profiles.py)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
})

FIGSIZE = (6, 4)
DPI = 200

# ---------------------------------------------------------------------------
# Tiny MLP on synthetic regression
# ---------------------------------------------------------------------------

def make_data(key, n=200, d_in=4):
    """Synthetic regression: y = sin(Wx).sum(axis=-1) + noise."""
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (n, d_in))
    W_true = jax.random.normal(k2, (d_in, 3)) * 0.5
    y = jnp.sin(X @ W_true).sum(axis=-1) + 0.1 * jax.random.normal(k3, (n,))
    return X, y


def init_mlp(key, d_in=4, d_hid=8, d_out=1):
    """Tiny 2-layer MLP: (d_in -> d_hid -> d_out)."""
    k1, k2 = jax.random.split(key)
    params = {
        "w1": jax.random.normal(k1, (d_in, d_hid)) * 0.5,
        "b1": jnp.zeros(d_hid),
        "w2": jax.random.normal(k2, (d_hid, d_out)) * 0.5,
        "b2": jnp.zeros(d_out),
    }
    return params


def mlp_forward(params, X):
    h = jnp.tanh(X @ params["w1"] + params["b1"])
    return (h @ params["w2"] + params["b2"]).squeeze(-1)


def loss_fn(params, X, y):
    return jnp.mean((mlp_forward(params, X) - y) ** 2)


# ---------------------------------------------------------------------------
# Experiment 1: Learnable Scalar
# ---------------------------------------------------------------------------

def run_scalar(n_steps=500):
    """Train with learnable scalar and log s at each step."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X, y = make_data(k1)
    params = init_mlp(k2)

    opt = custom_sgd_learnable_scalar(
        learning_rate=0.05,
        momentum=0.9,
        xi=0.5,
        beta=0.8,
        metric_lr=0.01,
        metric_reg=1e-3,
        metric_clip=4.0,
    )
    state = opt.init(params)

    grad_fn = jax.jit(jax.grad(loss_fn))

    steps = []
    s_vals = []
    losses = []

    for t in range(n_steps):
        grads = grad_fn(params, X, y)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        s_val = float(state.log_scale)
        l_val = float(loss_fn(params, X, y))
        steps.append(t)
        s_vals.append(s_val)
        losses.append(l_val)

    return np.array(steps), np.array(s_vals), np.array(losses)


# ---------------------------------------------------------------------------
# Experiment 2: Learnable Diagonal
# ---------------------------------------------------------------------------

def run_diag(n_steps=500):
    """Train with learnable diagonal and log selected s_i at each step."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X, y = make_data(k1)
    params = init_mlp(k2)

    opt = custom_sgd_learnable_diag(
        learning_rate=0.05,
        momentum=0.9,
        xi=0.5,
        beta=0.8,
        metric_lr=0.01,
        metric_reg=1e-3,
        metric_clip=4.0,
    )
    state = opt.init(params)

    grad_fn = jax.jit(jax.grad(loss_fn))

    # We will track a few hand-picked s_i values from different parameter leaves.
    # log_diag has the same pytree structure as params: {w1, b1, w2, b2}.
    # Pick indices that sample different layers / roles.
    tracked = [
        ("w1", (0, 0)),
        ("w1", (1, 3)),
        ("w1", (3, 7)),
        ("b1", (0,)),
        ("b1", (4,)),
        ("w2", (0, 0)),
        ("w2", (4, 0)),
        ("w2", (7, 0)),
        ("b2", (0,)),
    ]

    steps = []
    s_history = {f"{name}{idx}": [] for name, idx in tracked}
    losses = []

    for t in range(n_steps):
        grads = grad_fn(params, X, y)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        for name, idx in tracked:
            val = float(state.log_diag[name][idx])
            s_history[f"{name}{idx}"].append(val)

        steps.append(t)
        losses.append(float(loss_fn(params, X, y)))

    return np.array(steps), s_history, np.array(losses)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scalar(steps, s_vals, losses):
    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    color_s = "#d62728"
    color_loss = "#1f77b4"

    ax1.plot(steps, s_vals, color=color_s, label=r"$s$")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$s$", color=color_s)
    ax1.tick_params(axis="y", which="both", labelcolor=color_s, color=color_s)
    ax1.spines["left"].set_color(color_s)

    ax2 = ax1.twinx()
    ax2.plot(steps, losses, color=color_loss, alpha=0.5, linewidth=1.2, label="Loss")
    ax2.set_ylabel(r"Loss", color=color_loss)
    ax2.tick_params(axis="y", which="both", labelcolor=color_loss, color=color_loss)
    ax2.spines["right"].set_color(color_loss)
    ax2.set_yscale("log")

    ax1.set_title(r"Learnable scalar metric: evolution of $s$")
    ax1.grid(True)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "s_evolution_scalar.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path}")


def plot_diag(steps, s_history, losses):
    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # Readable labels for the tracked components
    label_map = {
        "w1(0, 0)": r"$s_{{w_1}}^{{(0,0)}}$",
        "w1(1, 3)": r"$s_{{w_1}}^{{(1,3)}}$",
        "w1(3, 7)": r"$s_{{w_1}}^{{(3,7)}}$",
        "b1(0,)": r"$s_{{b_1}}^{{(0)}}$",
        "b1(4,)": r"$s_{{b_1}}^{{(4)}}$",
        "w2(0, 0)": r"$s_{{w_2}}^{{(0,0)}}$",
        "w2(4, 0)": r"$s_{{w_2}}^{{(4,0)}}$",
        "w2(7, 0)": r"$s_{{w_2}}^{{(7,0)}}$",
        "b2(0,)": r"$s_{{b_2}}^{{(0)}}$",
    }

    # Use a qualitative colormap
    cmap = plt.cm.tab10
    for i, (key, vals) in enumerate(s_history.items()):
        lbl = label_map.get(key, key)
        ax1.plot(steps, vals, color=cmap(i / len(s_history)), label=lbl)

    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$s_i$")
    ax1.set_title(r"Learnable diagonal metric: evolution of $s_i$")
    ax1.legend(fontsize=7, ncol=3, loc="upper right")
    ax1.grid(True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "s_evolution_diag.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating s-evolution plots ...")

    print("  Running learnable scalar experiment ...")
    steps_s, s_vals, losses_s = run_scalar(n_steps=500)
    plot_scalar(steps_s, s_vals, losses_s)

    print("  Running learnable diagonal experiment ...")
    steps_d, s_hist, losses_d = run_diag(n_steps=500)
    plot_diag(steps_d, s_hist, losses_d)

    print("Done.")
