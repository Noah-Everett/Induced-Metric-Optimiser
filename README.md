# Induced-Metric-Optimiser

**Figure 1.** _The projection of a curve to the loss landscape from the higher dimensional space. When accounting for the higher dimensional space, the distance travelled is significantly further in highly curved regions._
<img src="images/projection.png" alt="Figure 1" width="600"/>

This repository contains the experiments and implementation for induced metric optimisers. These are preconditioning-based optimisers that utilise the induced metric, which is often illustrated in graphical visualisations of loss landscapes. Specifically, the approach involves pulling back the metric from a higher-dimensional space onto the loss landscape. This can be interpreted as a smoothed variant of gradient clipping.

Implementations in both JAX and PyTorch are included, along with a file demonstrating the implementations are equivalent. All experiments were performed in JAX.

For further details, see the paper: https://arxiv.org/abs/2509.03594

---
**Figure 2.** _The parameter update vs the gradient, where is a scale that depends on the local curvature._  
![Figure 2](images/grad_clip.png)

---

## Repository Structure

```
optimisers/          # Optimizer implementations (JAX + PyTorch)
parameters/          # Hyperparameter sweep scripts and shared utilities
analysis/            # Analysis notebooks for visualising results
results/             # Local sweep results (JSON format)
documents/           # Additional notebooks and notes
images/              # Figures
```

## Setup

### Dependencies

```bash
pip install jax jaxlib optax flax numpy matplotlib seaborn pandas
```

For local sweeps (no WandB required):
```bash
pip install optuna
```

For WandB sweeps:
```bash
pip install wandb
```

## Running Sweeps

All sweep scripts support two backends: **WandB** (cloud logging) and **local** (Optuna + JSON files). The local backend requires no account or API key.

### Local backend (default)

```bash
cd parameters

# Run a sweep for a single optimizer
python sweep_mnist_mlp.py --optimiser adam --num_runs 50 --backend local

# Specify run index (for organising multiple runs)
python sweep_mnist_mlp.py --optimiser sgd_metric --num_runs 50 --backend local --index 0

# Results are saved to results/<task>/<optimizer>/run_<index>/*.json
```

### WandB backend

```bash
cd parameters

python sweep_mnist_mlp.py --optimiser adam --num_runs 50 --backend wandb
```

### Available sweep scripts

| Script | Task | Model |
|--------|------|-------|
| `sweep_mnist_mlp.py` | MNIST classification | MLP (64 hidden) |
| `sweep_cifar10_resnet18.py` | CIFAR-10 classification | ResNet-18 |
| `sweep_regression.py` | Synthetic regression | MLP |
| `sweep_shake.py` | Shakespeare language modelling | MiniGPT |
| `sweep_small_examples.py` | 2D test functions (Beale, Rosenbrock, etc.) | N/A |

### Sweep CLI options

```
--optimiser         Optimizer name (required, see list below)
--num_runs          Number of sweep trials (default: 50)
--backend           "wandb" or "local" (default: "wandb")
--index             Run index for organising results (default: 0)
--val_freq          Validation frequency in epochs (default: 1)
--search            Search method: "bayes", "grid", "random" (default: "bayes", WandB only)
--results_dir       Base directory for local results (default: "results")
```

### Available optimizers

**Baselines:** `adam`, `adamw`, `sgd`, `muon`

**Fixed metric:** `sgd_metric`, `sgd_log_metric`, `sgd_rms`

**Learnable metric:** `sgd_learn_scalar`, `sgd_learn_scalar_log`, `sgd_learn_diag`, `sgd_learn_diag_log`

**Off-diagonal (all pairs of {0, l, m, θ}):** `sgd_offdiag_0_l`, `sgd_offdiag_l_m`, `sgd_offdiag_m_theta`, etc.

## Analysis Notebooks

Each analysis notebook can load results from either WandB or local JSON files. Configure the backend in the second code cell:

```python
# Backend: "wandb" or "local"
BACKEND = "local"

# Task and run index — must match your sweep
TASK_TAG = "mnist_mlp"
RUN_INDEX = 0

# Optimizers to analyse
OPTIMIZERS = ['adam', 'sgd', 'sgd_metric', 'sgd_log_metric']

# Results directory (for local backend)
RESULTS_DIR = os.path.join('..', 'results')
```

### Available notebooks

| Notebook | Task |
|----------|------|
| `analysis/mnist_analysis.ipynb` | MNIST MLP results |
| `analysis/cifar_analysis.ipynb` | CIFAR-10 ResNet-18 results |
| `analysis/regression_analysis.ipynb` | Regression results |
| `analysis/shake_analysis.ipynb` | Shakespeare results |
| `analysis/shake_analysis_full.ipynb` | Shakespeare full sweep (runs training inline) |
| `analysis/small_examples.ipynb` | 2D test function results |

### What the notebooks produce

- **2D histograms** — distribution of best accuracy vs epoch across sweep runs
- **Training curves** — loss and accuracy over wall time and epochs
- **Time-to-best** — wall time and epoch to reach best metric per optimizer
- **Speedrun analysis** — time to reach accuracy/loss targets
- **Hyperparameter tables** — best config for each optimizer (printed + saved to CSV)

## Local Results Format

Each sweep run saves a JSON file to `results/<task>/<optimizer>/run_<index>/<run_id>.json`:

```json
{
  "config": { "learning_rate": 0.01, "momentum": 0.9, ... },
  "history": [
    { "epoch": 0, "train_loss": 2.30, "train_acc": 0.10, ... },
    { "epoch": 1, "train_loss": 2.15, ... },
    ...
  ],
  "summary": {
    "final_max_val_acc": 0.977,
    "final_max_acc_epoch": 120,
    "sweep_metric": -0.977,
    "total_training_time_sec": 60.2,
    ...
  }
}
```

## Optimizer Implementations

Both JAX and PyTorch implementations are provided in `optimisers/`:

| File | Description |
|------|-------------|
| `jax_fixed.py` | Fixed induced metric (JAX) |
| `jax_learnable_scalar.py` | Learnable scalar metric (JAX) |
| `jax_learnable_diag.py` | Learnable diagonal metric (JAX) |
| `jax_offdiag.py` | Off-diagonal metric pairs (JAX) |
| `torch_*.py` | PyTorch equivalents |
| `compare_*.py` | Scripts verifying JAX/PyTorch equivalence |
