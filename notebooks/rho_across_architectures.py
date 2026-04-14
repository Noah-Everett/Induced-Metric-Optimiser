"""
IMO-143: Measure Hutchinson diagonal estimator rho across neural network architectures.

Tests the single-sample Hutchinson diagonal Hessian estimator on 4 architectures:
  1. Small CNN on CIFAR-10
  2. Mini ResNet on CIFAR-10
  3. Small Transformer (causal LM)
  4. Small LSTM (causal LM)

For each architecture, computes 10 independent Hutchinson diagonal estimates
using GLOBAL Rademacher vectors, then reports per-layer:
  - rho = sqrt(E[h^2]/E[h]^2 - 1)   (coefficient of variation)
  - neg_frac = fraction of estimates that are negative

Run: PYTHONPATH=repo .conda/bin/python3.11 repo/notebooks/rho_across_architectures.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import time
from typing import Sequence

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# 1. Small CNN on CIFAR-10
class SmallCNN(nn.Module):
    """3 conv layers + dense head for CIFAR-10 (10 classes)."""
    @nn.compact
    def __call__(self, x):
        # x: (batch, 32, 32, 3)
        x = nn.relu(nn.Conv(32, (3, 3), padding='SAME')(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.relu(nn.Conv(64, (3, 3), padding='SAME')(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.relu(nn.Conv(128, (3, 3), padding='SAME')(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(10)(x)
        return x


# 2. Mini ResNet on CIFAR-10
class ResBlock(nn.Module):
    filters: int
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.relu(nn.Conv(self.filters, (3, 3), strides=self.strides, padding='SAME')(x))
        y = nn.Conv(self.filters, (3, 3), padding='SAME')(y)
        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), strides=self.strides, padding='SAME')(residual)
        return nn.relu(y + residual)


class MiniResNet(nn.Module):
    """Simple ResNet with 4 residual blocks for CIFAR-10."""
    @nn.compact
    def __call__(self, x):
        # x: (batch, 32, 32, 3)
        x = nn.relu(nn.Conv(32, (3, 3), padding='SAME')(x))
        x = ResBlock(32)(x)
        x = ResBlock(64, strides=(2, 2))(x)
        x = ResBlock(64)(x)
        x = ResBlock(128, strides=(2, 2))(x)
        # Global average pool
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(10)(x)
        return x


# 3. Small Transformer (causal LM)
class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, mask):
        # Self-attention with causal mask
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=True,
        )(x, x, mask=mask)
        x = nn.LayerNorm()(x + attn_out)
        # MLP
        mlp_out = nn.Dense(self.mlp_dim)(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.embed_dim)(mlp_out)
        x = nn.LayerNorm()(x + mlp_out)
        return x


class SmallTransformer(nn.Module):
    vocab_size: int = 1000
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = 512
    max_len: int = 64

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len) integer tokens
        batch, seq_len = x.shape
        # Token + position embedding
        tok_emb = nn.Embed(self.vocab_size, self.embed_dim)(x)
        pos_emb = nn.Embed(self.max_len, self.embed_dim)(jnp.arange(seq_len))
        x = tok_emb + pos_emb[None, :, :]

        # Causal mask: (1, 1, seq, seq) — True means attend
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        causal_mask = causal_mask[None, None, :, :]  # broadcast over batch and heads

        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.mlp_dim)(x, causal_mask)

        logits = nn.Dense(self.vocab_size)(x)
        return logits


# 4. Small LSTM (causal LM)
class SmallLSTM(nn.Module):
    vocab_size: int = 1000
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len) integer tokens
        x = nn.Embed(self.vocab_size, self.embed_dim)(x)
        for i in range(self.num_layers):
            x = nn.RNN(nn.LSTMCell(self.hidden_dim))(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits, labels, num_classes):
    """Standard cross-entropy loss."""
    one_hot = jax.nn.one_hot(labels, num_classes)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1))


def next_token_loss(logits, targets, vocab_size):
    """Next-token prediction loss: predict token t+1 from position t."""
    # logits: (batch, seq, vocab), targets: (batch, seq)
    # Shift: predict targets[:, 1:] from logits[:, :-1]
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    one_hot = jax.nn.one_hot(targets, vocab_size)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1))


# ---------------------------------------------------------------------------
# Hutchinson diagonal estimator (global Rademacher vector)
# ---------------------------------------------------------------------------

def rademacher_like(rng_key, params):
    """Generate a GLOBAL Rademacher (+/-1) pytree matching params."""
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(rng_key, len(leaves))
    v_leaves = [
        2.0 * jax.random.bernoulli(k, 0.5, l.shape).astype(l.dtype) - 1.0
        for k, l in zip(keys, leaves)
    ]
    return treedef.unflatten(v_leaves)


def hutchinson_diag_global(loss_fn, params, rng_key):
    """Single global Hutchinson diagonal estimate.

    Uses a single Rademacher vector across ALL parameters (global estimate).
    h = v * HVP(v) where HVP(v) = d/dt grad(loss)(params + t*v)|_{t=0}
    """
    v = rademacher_like(rng_key, params)
    grad_fn = jax.grad(loss_fn)
    _, hvp = jax.jvp(grad_fn, (params,), (v,))
    # h_i = v_i * (Hv)_i — element-wise per leaf
    h_diag = jax.tree.map(lambda vi, hvpi: vi * hvpi, v, hvp)
    return h_diag


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_rho(model, loss_fn_builder, data_builder, num_samples=10, seed=0):
    """
    Measure rho and neg_frac for a model.

    Returns list of dicts, one per parameter leaf:
        {name, shape, count, rho, neg_frac}
    """
    key = jax.random.PRNGKey(seed)

    # Build data
    key, data_key = jax.random.split(key)
    data = data_builder(data_key)

    # Init model
    key, init_key = jax.random.split(key)
    if isinstance(data, tuple):
        params = model.init(init_key, data[0])
    else:
        params = model.init(init_key, data)

    # Build loss function (closed over data)
    loss_fn = loss_fn_builder(model, data)

    # Collect path info
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    leaf_names = [jax.tree_util.keystr(path) for path, _ in paths_and_leaves]
    leaf_shapes = [leaf.shape for _, leaf in paths_and_leaves]
    leaf_sizes = [leaf.size for _, leaf in paths_and_leaves]

    total_params = sum(leaf_sizes)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of parameter leaves: {len(leaf_names)}")

    # Collect Hutchinson estimates
    # We'll JIT the Hutchinson computation for speed
    @jax.jit
    def compute_one_sample(params, rng_key):
        return hutchinson_diag_global(loss_fn, params, rng_key)

    print(f"  Computing {num_samples} Hutchinson samples...")
    all_estimates = []  # list of lists-of-arrays, one per sample

    for i in range(num_samples):
        key, sample_key = jax.random.split(key)
        t0 = time.time()
        h_diag = compute_one_sample(params, sample_key)
        h_leaves = jax.tree.leaves(h_diag)
        # Block until computation is done
        _ = [l.block_until_ready() for l in h_leaves]
        dt = time.time() - t0
        all_estimates.append([np.array(l) for l in h_leaves])
        if i == 0:
            print(f"    Sample 0 took {dt:.2f}s (includes JIT compilation)")
        elif i == 1:
            print(f"    Sample 1 took {dt:.2f}s (post-JIT)")

    # Compute per-leaf rho and neg_frac
    results = []
    for j in range(len(leaf_names)):
        # Stack estimates for this leaf: (num_samples, *shape)
        stacked = np.stack([all_estimates[i][j] for i in range(num_samples)], axis=0)

        # Per-element statistics, then aggregate
        # E[h] and E[h^2] per element
        mean_h = np.mean(stacked, axis=0)          # (*shape,)
        mean_h2 = np.mean(stacked ** 2, axis=0)    # (*shape,)

        # rho per element: sqrt(E[h^2]/E[h]^2 - 1)
        # But E[h] can be near zero for some elements, causing division issues.
        # Use the aggregate formula: rho = std(h) / |mean(h)| averaged sensibly.
        #
        # Global rho for this leaf: treat all elements as one population
        # rho_leaf = sqrt( sum(Var[h_i]) / sum(E[h_i])^2 )
        #          = sqrt( (sum E[h_i^2] - sum E[h_i]^2) / (sum E[h_i])^2 )
        #
        # But this conflates elements. Better: element-wise rho, then median.
        # Actually, for the paper we care about the GLOBAL rho — the CV of
        # the sum of all h_i. But let's report both.

        # --- Element-wise rho (median over elements) ---
        denom = mean_h ** 2
        safe = np.abs(mean_h) > 1e-30
        elem_rho = np.full_like(mean_h, np.nan)
        ratio = np.where(safe, mean_h2 / np.where(safe, denom, 1.0), np.nan)
        valid = ratio > 1.0
        elem_rho = np.where(valid, np.sqrt(ratio - 1.0), np.where(safe, 0.0, np.nan))
        median_rho = float(np.nanmedian(elem_rho))

        # --- Leaf-aggregate rho ---
        # Treat sum(h_i) across elements as the estimand.
        # Var[sum h_i] = sum_i Var[h_i] + 2 sum_{i<j} Cov[h_i, h_j]
        # With global Rademacher, there ARE cross-element covariances.
        # Estimate directly from samples:
        leaf_sums = np.array([np.sum(all_estimates[i][j]) for i in range(num_samples)])
        leaf_mean = np.mean(leaf_sums)
        leaf_var = np.var(leaf_sums, ddof=1)
        if abs(leaf_mean) > 1e-30:
            leaf_rho = float(np.sqrt(leaf_var) / abs(leaf_mean))
        else:
            leaf_rho = float('inf')

        # --- neg_frac: fraction of (element, sample) pairs that are negative ---
        neg_frac = float(np.mean(stacked < 0))

        results.append({
            'name': leaf_names[j],
            'shape': leaf_shapes[j],
            'count': leaf_sizes[j],
            'median_rho': median_rho,
            'leaf_rho': leaf_rho,
            'neg_frac': neg_frac,
        })

    return results


# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------

def build_architectures():
    """Return list of (name, model, loss_fn_builder, data_builder) tuples."""
    batch_size = 32

    # --- 1. Small CNN ---
    def cnn_data(key):
        k1, k2 = jax.random.split(key)
        images = jax.random.normal(k1, (batch_size, 32, 32, 3))
        labels = jax.random.randint(k2, (batch_size,), 0, 10)
        return images, labels

    def cnn_loss_builder(model, data):
        images, labels = data
        def loss_fn(params):
            logits = model.apply(params, images)
            return cross_entropy_loss(logits, labels, 10)
        return loss_fn

    # --- 2. Mini ResNet ---
    def resnet_data(key):
        k1, k2 = jax.random.split(key)
        images = jax.random.normal(k1, (batch_size, 32, 32, 3))
        labels = jax.random.randint(k2, (batch_size,), 0, 10)
        return images, labels

    def resnet_loss_builder(model, data):
        images, labels = data
        def loss_fn(params):
            logits = model.apply(params, images)
            return cross_entropy_loss(logits, labels, 10)
        return loss_fn

    # --- 3. Small Transformer ---
    def transformer_data(key):
        tokens = jax.random.randint(key, (batch_size, 64), 0, 1000)
        return tokens

    def transformer_loss_builder(model, data):
        tokens = data
        def loss_fn(params):
            logits = model.apply(params, tokens)
            return next_token_loss(logits, tokens, 1000)
        return loss_fn

    # --- 4. Small LSTM ---
    def lstm_data(key):
        tokens = jax.random.randint(key, (batch_size, 64), 0, 1000)
        return tokens

    def lstm_loss_builder(model, data):
        tokens = data
        def loss_fn(params):
            logits = model.apply(params, tokens)
            return next_token_loss(logits, tokens, 1000)
        return loss_fn

    return [
        ("Small CNN (CIFAR-10)",     SmallCNN(),       cnn_loss_builder,         cnn_data),
        ("Mini ResNet (CIFAR-10)",   MiniResNet(),     resnet_loss_builder,      resnet_data),
        ("Small Transformer (LM)",   SmallTransformer(), transformer_loss_builder, transformer_data),
        ("Small LSTM (LM)",          SmallLSTM(),      lstm_loss_builder,        lstm_data),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

NUM_SAMPLES = 10

if __name__ == "__main__":
    print("=" * 100)
    print("IMO-143: Hutchinson diagonal estimator rho across architectures")
    print("=" * 100)
    print(f"Samples per architecture: {NUM_SAMPLES}")
    print(f"Global Rademacher vectors (single vector across all params)\n")

    architectures = build_architectures()
    all_results = {}

    for arch_name, model, loss_builder, data_builder in architectures:
        print(f"\n{'─' * 100}")
        print(f"  Architecture: {arch_name}")
        print(f"{'─' * 100}")

        t0 = time.time()
        results = measure_rho(model, loss_builder, data_builder,
                              num_samples=NUM_SAMPLES, seed=42)
        dt = time.time() - t0
        print(f"  Total time: {dt:.1f}s")

        all_results[arch_name] = results

        # Print per-layer table
        print(f"\n  {'Layer':55s} {'Params':>8s}  {'med rho':>10s}  {'leaf rho':>10s}  {'neg_frac':>8s}")
        print(f"  {'─'*55}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
        for r in results:
            rho_str = f"{r['median_rho']:.1f}" if np.isfinite(r['median_rho']) else "inf"
            lrho_str = f"{r['leaf_rho']:.1f}" if np.isfinite(r['leaf_rho']) else "inf"
            print(f"  {r['name']:55s} {r['count']:>8d}  {rho_str:>10s}  {lrho_str:>10s}  {r['neg_frac']:>8.3f}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 100}")
    print("SUMMARY: Per-architecture aggregates")
    print(f"{'=' * 100}")
    print(f"  {'Architecture':30s}  {'Params':>10s}  {'Leaves':>6s}  "
          f"{'med(rho)':>10s}  {'max(rho)':>10s}  "
          f"{'mean neg%':>10s}  {'Verdict':>10s}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for arch_name, results in all_results.items():
        total_params = sum(r['count'] for r in results)
        n_leaves = len(results)

        rhos = [r['median_rho'] for r in results if np.isfinite(r['median_rho'])]
        leaf_rhos = [r['leaf_rho'] for r in results if np.isfinite(r['leaf_rho'])]

        med_rho = float(np.median(rhos)) if rhos else float('inf')
        max_rho = float(np.max(rhos)) if rhos else float('inf')

        mean_neg = float(np.mean([r['neg_frac'] for r in results]))

        if med_rho > 10:
            verdict = "BROKEN"
        elif med_rho > 3:
            verdict = "NOISY"
        else:
            verdict = "OK"

        med_str = f"{med_rho:.1f}" if np.isfinite(med_rho) else "inf"
        max_str = f"{max_rho:.1f}" if np.isfinite(max_rho) else "inf"

        print(f"  {arch_name:30s}  {total_params:>10,d}  {n_leaves:>6d}  "
              f"{med_str:>10s}  {max_str:>10s}  "
              f"{100*mean_neg:>9.1f}%  {verdict:>10s}")

    # ---------------------------------------------------------------------------
    # Dense vs Conv breakdown
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 100}")
    print("BREAKDOWN: Dense vs Conv vs Attention vs Embedding vs LSTM layers")
    print(f"{'=' * 100}")

    categories = {
        'Conv': lambda n: 'Conv_' in n and 'kernel' in n,
        'Dense': lambda n: 'Dense' in n and 'kernel' in n,
        'Attention': lambda n: ('query' in n or 'key' in n.lower().split("'")[-1]
                                 or 'value' in n or 'out' in n)
                               and 'Attention' in n and 'kernel' in n,
        'Embed': lambda n: 'Embed' in n or 'embedding' in n.lower(),
        'LSTM': lambda n: 'LSTM' in n or 'lstm' in n,
        'LayerNorm': lambda n: 'LayerNorm' in n,
        'Bias': lambda n: 'bias' in n,
    }

    print(f"  {'Category':15s}  {'Count':>6s}  {'med(rho)':>10s}  {'max(rho)':>10s}  {'mean neg%':>10s}")
    print(f"  {'─'*15}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")

    for cat_name, cat_fn in categories.items():
        matching = []
        for arch_name, results in all_results.items():
            for r in results:
                if cat_fn(r['name']):
                    matching.append(r)
        if not matching:
            continue

        rhos = [r['median_rho'] for r in matching if np.isfinite(r['median_rho'])]
        med_rho = float(np.median(rhos)) if rhos else float('inf')
        max_rho = float(np.max(rhos)) if rhos else float('inf')
        mean_neg = float(np.mean([r['neg_frac'] for r in matching]))

        med_str = f"{med_rho:.1f}" if np.isfinite(med_rho) else "inf"
        max_str = f"{max_rho:.1f}" if np.isfinite(max_rho) else "inf"

        print(f"  {cat_name:15s}  {len(matching):>6d}  {med_str:>10s}  {max_str:>10s}  {100*mean_neg:>9.1f}%")

    print(f"\n{'=' * 100}")
    print("Key: rho = coefficient of variation (sqrt(Var/mean^2))")
    print("     rho >> 1 means the Hutchinson estimate is noise-dominated")
    print("     neg_frac ~ 0.5 means estimates are as likely negative as positive")
    print("     BROKEN = med(rho) > 10, NOISY = med(rho) > 3, OK = med(rho) <= 3")
    print(f"{'=' * 100}")
