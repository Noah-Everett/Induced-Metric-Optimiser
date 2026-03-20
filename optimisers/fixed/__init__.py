"""
Fixed induced-metric SGD optimizers (JAX + PyTorch).

Public API::

    from optimisers.fixed import custom_sgd, custom_sgd_log, custom_sgd_rms
    from optimisers.fixed import CustomSGD, CustomSGDLog, CustomSGDRMS
"""

from .jax import custom_sgd, custom_sgd_log, custom_sgd_rms, SGDState
from .torch import CustomSGD, CustomSGDLog, CustomSGDRMS

__all__ = [
    # JAX
    'custom_sgd', 'custom_sgd_log', 'custom_sgd_rms', 'SGDState',
    # PyTorch
    'CustomSGD', 'CustomSGDLog', 'CustomSGDRMS',
]
