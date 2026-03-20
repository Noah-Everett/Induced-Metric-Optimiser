"""
Learnable diagonal induced-metric SGD optimizers (JAX + PyTorch).

Public API::

    from optimisers.learnable_diag import custom_sgd_learnable_diag, custom_sgd_log_learnable_diag
    from optimisers.learnable_diag import SGDLearnableDiag
"""

from .jax import (
    custom_sgd_learnable_diag,
    custom_sgd_log_learnable_diag,
    SGDLearnableDiagState,
)
from .torch import SGDLearnableDiag

__all__ = [
    # JAX
    'custom_sgd_learnable_diag', 'custom_sgd_log_learnable_diag',
    'SGDLearnableDiagState',
    # PyTorch
    'SGDLearnableDiag',
]
