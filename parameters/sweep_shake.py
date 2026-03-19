"""
Tiny Shakespeare MiniGPT hyperparameter sweep.

Usage::

    python sweep_shake.py --optimiser adam --num_runs 50 --backend wandb
    python sweep_shake.py --optimiser sgd_learn_scalar --num_runs 100 --backend local
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
import requests

from shared_models import MiniGPT
from optimizer_registry import create_optimizer, needs_loss
from sweep_utils import SweepRunner, setup_argparser

# Parse CLI
parser = setup_argparser("Tiny Shakespeare MiniGPT Hyperparameter Sweep")
args = parser.parse_args()

# Fixed architecture configuration
ARCHITECTURE_CONFIG = {
    "seq_len": 256,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "dropout_rate": 0.1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_shakespeare():
    """Download the tiny Shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "tinyshakespeare.txt"

    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)

    return filename


def load_shakespeare(seq_len=128, val_split=0.1):
    """Load and preprocess the tiny Shakespeare dataset for character-level modeling."""
    filename = download_shakespeare()

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    data = jnp.array([char_to_idx[ch] for ch in text])

    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data, vocab_size


def create_text_batches(data, seq_len, batch_size, seed):
    """Create batches for language modeling (input, target) pairs."""
    key = jax.random.PRNGKey(seed)

    num_sequences = (len(data) - 1) // seq_len

    inputs = []
    targets = []

    for i in range(num_sequences):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        if end_idx < len(data):
            inputs.append(data[start_idx:end_idx])
            targets.append(data[start_idx + 1:end_idx + 1])

    inputs = jnp.array(inputs)
    targets = jnp.array(targets)

    perm = jax.random.permutation(key, len(inputs))
    inputs = inputs[perm]
    targets = targets[perm]

    num_batches = len(inputs) // batch_size
    batches = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batches.append((inputs[start_idx:end_idx], targets[start_idx:end_idx]))

    return batches


# ---------------------------------------------------------------------------
# Loss / perplexity helpers
# ---------------------------------------------------------------------------

def loss_fn(params, x, y, model, key):
    """Compute cross-entropy loss for language modeling."""
    logits = model.apply(params, x, train=True, rngs={"dropout": key})
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = y.reshape(-1)
    return optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()


@partial(jax.jit, static_argnums=(2,))
def _val_loss_batch(params, batch, model):
    """JIT-compiled loss for a single validation batch."""
    x, y = batch
    logits = model.apply(params, x, train=False)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = y.reshape(-1)
    return optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean(), y.size


def perplexity_fn(params, data_batches, model):
    """Compute perplexity over all batches."""
    total_loss = 0.0
    total_tokens = 0

    for batch in data_batches:
        batch_loss, n_tokens = _val_loss_batch(params, batch, model)
        total_loss += batch_loss * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    return jnp.exp(avg_loss)


# ---------------------------------------------------------------------------
# Generic training function
# ---------------------------------------------------------------------------

def train(config, seed, logger):
    # Merge fixed architecture config
    full_config = {**ARCHITECTURE_CONFIG, **config}

    seq_len = full_config["seq_len"]
    embed_dim = full_config["embed_dim"]
    num_heads = full_config["num_heads"]
    num_layers = full_config["num_layers"]
    dropout_rate = full_config["dropout_rate"]
    batch_size = full_config.get("batch_size", 256)
    n_epochs = full_config.get("n_epochs", 250)

    train_data, val_data, vocab_size = load_shakespeare(seq_len=seq_len, val_split=0.1)

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        max_seq_len=seq_len,
    )

    train_batches = create_text_batches(train_data, seq_len, batch_size, seed)
    val_batches = create_text_batches(val_data, seq_len, batch_size, seed + 1)

    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    init_key, dropout_key = jax.random.split(key, 2)
    params = model.init({"params": init_key, "dropout": dropout_key}, dummy_input, train=True)

    optimizer = create_optimizer(args.optimiser, full_config)
    opt_state = optimizer.init(params)
    use_loss = needs_loss(args.optimiser)

    if use_loss:
        @jax.jit
        def train_step(params, opt_state, x, y, dropout_key):
            def loss_grad_fn(p):
                loss = loss_fn(p, x, y, model, dropout_key)
                return loss, loss

            (loss, loss_for_opt), grads = jax.value_and_grad(loss_grad_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, loss_for_opt, params)
            return optax.apply_updates(params, updates), opt_state_new, loss
    else:
        @jax.jit
        def train_step(params, opt_state, x, y, dropout_key):
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, model, dropout_key))(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state_new, loss

    # Pre-generate batch keys for all epochs
    batch_keys_per_epoch = []
    for epoch in range(n_epochs):
        epoch_key = jax.random.PRNGKey(seed + epoch)
        batch_keys = [jax.random.fold_in(epoch_key, i) for i in range(len(train_batches))]
        batch_keys_per_epoch.append(batch_keys)

    min_val_perplexity = float("inf")
    min_perp_epoch = 0
    train_time = 0.0
    pruned = False

    for epoch in range(n_epochs):
        epoch_losses = []

        epoch_start = time.time()
        for i, (x_batch, y_batch) in enumerate(train_batches):
            batch_key = batch_keys_per_epoch[epoch][i]
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch, batch_key)
            epoch_losses.append(loss)
        train_time += time.time() - epoch_start

        avg_train_loss = float(jnp.mean(jnp.array(epoch_losses)))

        if epoch % args.val_freq == 0 or epoch == n_epochs - 1:
            val_perplexity = float(perplexity_fn(params, val_batches, model))
            if val_perplexity < min_val_perplexity:
                min_val_perplexity = val_perplexity
                min_perp_epoch = epoch

            logger.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_perplexity": val_perplexity,
                "train_time_seconds": train_time,
            })

            # Check for pruning (minimize perplexity)
            if logger.report_and_check_prune(epoch, val_perplexity):
                pruned = True
                break
        else:
            logger.log({"epoch": epoch, "train_loss": avg_train_loss})

    return {
        "objective": min_val_perplexity,
        "summary": {
            "final_min_val_perplexity": min_val_perplexity,
            "final_min_perp_epoch": min_perp_epoch,
            "sweep_metric": min_val_perplexity,
            "pruned": pruned,
            "architecture": "MiniGPT",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_fixed_params = {
        "batch_size": {"values": [args.batch_size if args.batch_size else 256]},
        "n_epochs": {"value": 250},
    }

    runner = SweepRunner(
        backend=args.backend,
        project="induced_metric",
        task_tag="shakespeare_minigpt",
        optimizer_name=args.optimiser,
        args=args,
        task_fixed_params=task_fixed_params,
        results_dir=args.results_dir,
    )
    runner.run(train)
