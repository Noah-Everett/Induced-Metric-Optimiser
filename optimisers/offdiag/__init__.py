"""
Off-diagonal induced-metric SGD optimizers (JAX + PyTorch).

Public API::

    from optimisers.offdiag import custom_sgd_offdiag
    from optimisers.offdiag import CustomSGDOffDiag
"""

from .jax import custom_sgd_offdiag, OffDiagSGDState
from .torch import CustomSGDOffDiag

__all__ = [
    # JAX
    'custom_sgd_offdiag', 'OffDiagSGDState',
    # PyTorch
    'CustomSGDOffDiag',
]
