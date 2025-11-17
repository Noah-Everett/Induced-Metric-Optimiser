#!/usr/bin/env python3
"""
Basic smoke test for the *learnable metric* optimizers.

It runs a tiny binary classification task on synthetic data for a few steps and
prints:
  - initial/final loss and accuracy
  - norm of the learned metric parameters s (log_diag) before/after

Supports:
  - PyTorch: SGDLearnableDiag (plain or log-loss)
  - JAX:     custom_sgd_learnable_diag / custom_sgd_log_learnable_diag

Usage examples:
  python analysis/learnable_metric_basic_test.py --framework torch
  python analysis/learnable_metric_basic_test.py --framework jax --log-loss
  python analysis/learnable_metric_basic_test.py --framework both --steps 300

Note:
  You mentioned you've renamed:
    torch_imp  -> torch_fixed
    jax_imp    -> jax_fixed
    compare    -> compare_fixed
  This script only tests the *learnable* variants you added; it doesn’t touch
  the fixed optimizers. (We keep imports fully namespaced to avoid confusion.)
"""

import argparse
import math
import sys
import time
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# -------------------- CLI --------------------

def build_args():
    p = argparse.ArgumentParser(description="Smoke test: learnable metric optimizers")
    p.add_argument("--framework", type=str, choices=["torch", "jax", "both"], default="both",
                   help="Which framework(s) to test.")
    p.add_argument("--log-loss", action="store_true", help="Use the log-loss embedding variant.")
    p.add_argument("--steps", type=int, default=200, help="Number of training steps.")
    p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--lr", type=float, default=1e-2, help="Base learning rate.")
    p.add_argument("--metric-lr", type=float, default=1e-3, help="Learning rate for metric parameters s.")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    p.add_argument("--beta", type=float, default=0.8, help="EMA beta for denominator.")
    p.add_argument("--xi", type=float, default=0.1, help="xi factor for v = xi * g^T gamma^{-1} g.")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="Decoupled weight decay.")
    return p.parse_args()

# -------------------- Synthetic data --------------------

def make_synth_binary(n_samples=4096, d=16, seed=0):
    rng = np.random.RandomState(seed)
    # Two Gaussians with slightly shifted means
    x_pos = rng.randn(n_samples // 2, d) + 0.75
    x_neg = rng.randn(n_samples // 2, d) - 0.75
    x = np.vstack([x_pos, x_neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)]).astype(np.int32)
    # Shuffle
    perm = rng.permutation(n_samples)
    return x[perm], y[perm]

# -------------------- Torch test --------------------

def run_torch(args):
    try:
        import torch
        from torch import nn
        from optimisers.torch_learnable import SGDLearnableDiag
    except Exception as e:
        print("[torch] Skipping: could not import torch or SGDLearnableDiag:", e, file=sys.stderr)
        return

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Data
    x, y = make_synth_binary(n_samples=4096, d=16, seed=args.seed)
    x = torch.tensor(x, device=device)
    y = torch.tensor(y, device=device).float()

    # Simple model: single hidden layer MLP -> 1 logit
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    ).to(device)

    # Optimizer
    opt = SGDLearnableDiag(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        xi=args.xi,
        beta=args.beta,
        weight_decay=args.weight_decay,
        metric_lr=args.metric_lr,
        metric_reg=1e-4,
        metric_clip=4.0,
        log_loss=args.log_loss,
    )

    bsz = args.batch_size
    num_steps = args.steps
    n = x.shape[0]

    bce = nn.BCEWithLogitsLoss()

    # helpers to inspect metric s-norm (sum of per-param Frobenius norms)
    def metric_norm():
        total = 0.0
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p, None)
                if st is None:
                    continue
                s = st.get("log_diag", None)
                if s is not None:
                    total += float(torch.norm(s).item())
        return total

    # initial metrics
    with torch.no_grad():
        logits = model(x)
        loss0 = bce(logits.squeeze(-1), y)
        acc0 = ((logits.squeeze(-1) > 0) == (y > 0.5)).float().mean().item()
    s0 = metric_norm()

    print("\n[torch] --- Learnable metric test ---")
    print(f"[torch] log_loss={args.log_loss}  steps={num_steps}  bsz={bsz}")
    print(f"[torch] initial: loss={loss0.item():.4f}  acc={acc0*100:.2f}%  ||s||={s0:.4f}")

    # training
    model.train()
    start = time.time()
    for t in range(num_steps):
        idx = slice((t * bsz) % n, ((t * bsz) % n) + bsz)
        xb = x[idx]
        yb = y[idx]

        opt.zero_grad(set_to_none=True)
        logits = model(xb).squeeze(-1)
        loss = bce(logits, yb)

        loss.backward()

        if args.log_loss:
            opt.step(loss=float(loss.item()))
        else:
            opt.step()

    elapsed = time.time() - start

    # final metrics
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        loss1 = bce(logits, y)
        acc1 = ((logits > 0) == (y > 0.5)).float().mean().item()
    s1 = metric_norm()

    print(f"[torch] final:   loss={loss1.item():.4f}  acc={acc1*100:.2f}%  ||s||={s1:.4f}  (Δ||s||={s1-s0:.4f})")
    print(f"[torch] time: {elapsed:.2f}s\n")

# -------------------- JAX test --------------------

def run_jax(args):
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from optimisers.jax_learnable import (
            custom_sgd_learnable_diag,
            custom_sgd_log_learnable_diag,
        )
    except Exception as e:
        print("[jax] Skipping: could not import jax/optax or jax_learnable:", e, file=sys.stderr)
        return

    key = jax.random.PRNGKey(args.seed)

    # Data
    x_np, y_np = make_synth_binary(n_samples=4096, d=16, seed=args.seed)
    x = jnp.array(x_np)
    y = jnp.array(y_np)

    # Model params (MLP 16->32->1)
    def init_params(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        W1 = jax.random.normal(k1, (16, 32)) * 0.1
        b1 = jnp.zeros((32,))
        W2 = jax.random.normal(k2, (32, 1)) * 0.1
        b2 = jnp.zeros((1,))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def forward(params, xb):
        h = jnp.tanh(jnp.dot(xb, params["W1"]) + params["b1"])
        logits = jnp.dot(h, params["W2"]) + params["b2"]  # shape (B, 1)
        return logits.squeeze(-1)

    def bce_with_logits(logits, targets):
        # mean BCE
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))

    params = init_params(key)

    # Optimizer
    if args.log_loss:
        opt = custom_sgd_log_learnable_diag(
            learning_rate=args.lr,
            momentum=args.momentum,
            xi=args.xi,
            beta=args.beta,
            weight_decay=args.weight_decay,
            metric_lr=args.metric_lr,
            metric_reg=1e-4,
            metric_clip=4.0,
        )
    else:
        opt = custom_sgd_learnable_diag(
            learning_rate=args.lr,
            momentum=args.momentum,
            xi=args.xi,
            beta=args.beta,
            weight_decay=args.weight_decay,
            metric_lr=args.metric_lr,
            metric_reg=1e-4,
            metric_clip=4.0,
        )
    state = opt.init(params)

    n = x.shape[0]
    bsz = args.batch_size
    num_steps = args.steps

    # metrics: compute loss/acc on full data
    def full_metrics(p):
        logits = forward(p, x)
        loss = bce_with_logits(logits, y)
        acc = jnp.mean((logits > 0) == (y > 0.5))
        return float(loss), float(acc)

    # norm of s (sum of per-leaf frobenius norms)
    def s_norm(st):
        def leaf_norm(s):
            return jnp.linalg.norm(s)
        parts = jax.tree_util.tree_map(leaf_norm, st.log_diag)
        return float(jax.tree_util.tree_reduce(lambda a, b: a + b, parts, 0.0))

    loss0, acc0 = full_metrics(params)
    s0 = s_norm(state)

    print("\n[jax] --- Learnable metric test ---")
    print(f"[jax] log_loss={args.log_loss}  steps={num_steps}  bsz={bsz}")
    print(f"[jax] initial: loss={loss0:.4f}  acc={acc0*100:.2f}%  ||s||={s0:.4f}")

    @jax.jit
    def step_fun(p, st, xb, yb):
        def loss_fn(pp):
            logits = forward(pp, xb)
            return bce_with_logits(logits, yb)
        if args.log_loss:
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, st2 = opt.update(grads, st, loss=loss, params=p)
        else:
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, st2 = opt.update(grads, st, params=p)
        p2 = optax.apply_updates(p, updates)
        return p2, st2, loss

    # Training loop
    start = time.time()
    idx = jnp.arange(n)
    for t in range(num_steps):
        # simple wrap-around minibatching
        lo = (t * bsz) % n
        hi = lo + bsz
        xb = x[lo:hi]
        yb = y[lo:hi]
        params, state, _ = step_fun(params, state, xb, yb)
    elapsed = time.time() - start

    loss1, acc1 = full_metrics(params)
    s1 = s_norm(state)

    print(f"[jax] final:   loss={loss1:.4f}  acc={acc1*100:.2f}%  ||s||={s1:.4f}  (Δ||s||={s1-s0:.4f})")
    print(f"[jax] time: {elapsed:.2f}s\n")

# -------------------- main --------------------

def main():
    args = build_args()
    if args.framework in ("torch", "both"):
        run_torch(args)
    if args.framework in ("jax", "both"):
        run_jax(args)

if __name__ == "__main__":
    main()
