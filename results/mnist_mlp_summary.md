# MNIST MLP Hyperparameter Sweep Results

## Summary

| Rank | Optimizer | Best Val Acc | Mean Val Acc | Trials |
|------|-----------|-------------|-------------|--------|
| 1 | sgd_learn_diag | 0.9799 | 0.5906 | 5 |
| 2 | sgd_metric | 0.9798 | 0.6473 | 5 |
| 3 | adam | 0.9769 | 0.9522 | 5 |
| 4 | sgd | 0.9766 | 0.6551 | 5 |
| 5 | sgd_learn_scalar | 0.9766 | 0.6561 | 5 |
| 6 | sgd_log_metric | 0.9754 | 0.5842 | 5 |

### Failed Optimizers
- **sgd_offdiag_l_m**: JAX ConcretizationTypeError in `jax_offdiag.py:89` — `jnp.split` called with traced (non-concrete) split indices inside `@jit`. The `_flat_to_tree` function needs to use static shapes.
- **sgd_offdiag_m_l**: Same error as sgd_offdiag_l_m.

## Best Hyperparameters per Optimizer

### sgd_learn_diag (best val acc: 0.979926)

- **beta**: 0.5100864022049432
- **learning_rate**: 1.156732719914599
- **metric_clip**: 1.8493564427131046
- **metric_lr**: 0.7072114131472232
- **metric_reg**: 0.002136832907235877
- **momentum**: 0.5951038616257767
- **n_epochs**: 200
- **xi**: 0.679657809075816
- All trial accs: [0.9540, 0.0800, 0.9799, 0.7409, 0.1981]

### sgd_metric (best val acc: 0.979818)

- **beta**: 0.5898682098281826
- **learning_rate**: 0.6715811311069951
- **momentum**: 0.2102157195714934
- **n_epochs**: 200
- **xi**: 0.005337032762603957
- All trial accs: [0.1051, 0.7109, 0.9798, 0.4837, 0.9570]

### adam (best val acc: 0.976888)

- **beta1**: 0.15445845403801214
- **beta2**: 0.831183304615207
- **eps**: 1e-08
- **learning_rate**: 0.015509913987594319
- **n_epochs**: 200
- All trial accs: [0.9747, 0.9763, 0.8574, 0.9769, 0.9755]

### sgd (best val acc: 0.976562)

- **learning_rate**: 0.1330324510152292
- **momentum**: 0.5926718993550663
- **n_epochs**: 200
- All trial accs: [0.9741, 0.9421, 0.1755, 0.2076, 0.9766]

### sgd_learn_scalar (best val acc: 0.976562)

- **beta**: 0.5100864022049432
- **learning_rate**: 1.156732719914599
- **metric_clip**: 1.8493564427131046
- **metric_lr**: 0.7072114131472232
- **metric_reg**: 0.002136832907235877
- **momentum**: 0.5951038616257767
- **n_epochs**: 200
- **xi**: 0.679657809075816
- All trial accs: [0.1585, 0.4239, 0.7699, 0.9766, 0.9518]

### sgd_log_metric (best val acc: 0.975369)

- **beta**: 0.9752558275593772
- **learning_rate**: 0.016136341713591334
- **momentum**: 0.700991852018085
- **n_epochs**: 200
- **xi**: 0.0012087541473056963
- All trial accs: [0.1162, 0.2912, 0.5656, 0.9728, 0.9754]

## Notes

- All runs used MNIST with a 2-layer MLP (64 hidden units, GELU activation)
- 5 Optuna trials per optimizer (random search)
- Validation every 10 epochs
- Training on CPU (JAX)
- Date: 2026-01-30
