"""
Learnable scalar induced-metric SGD optimizers (JAX + PyTorch).

Public API::

    from optimisers.learnable_scalar import custom_sgd_learnable_scalar, custom_sgd_log_learnable_scalar
    from optimisers.learnable_scalar import SGDLearnableScalar
"""

from .jax import (
    custom_sgd_learnable_scalar,
    custom_sgd_log_learnable_scalar,
    SGDLearnableScalarState,
)
from .torch import SGDLearnableScalar

__all__ = [
    # JAX
    'custom_sgd_learnable_scalar', 'custom_sgd_log_learnable_scalar',
    'SGDLearnableScalarState',
    # PyTorch
    'SGDLearnableScalar',
]
