"""
Basin-filling meta-optimizer for exploring loss landscapes.

Public API::

    from basin_filling import BasinFillingConfig, run_basin_filling
    from basin_filling.strategies import detection, walkthrough, radius

    config = BasinFillingConfig(
        detect_stuck=detection.loss_plateau(threshold=1e-6),
        walkthrough=walkthrough.entry(step_size=0.1),
        estimate_radius=radius.fixed(value=2.0),
    )
    result = run_basin_filling(loss_fn, grad_fn, init, create_opt, config)
"""

from .core import (
    Basin,
    StepRecord,
    StepInfo,
    BasinFillingConfig,
    run_basin_filling,
    run_random_restarts,
)
from . import strategies

__all__ = [
    'Basin', 'StepRecord', 'StepInfo', 'BasinFillingConfig',
    'run_basin_filling', 'run_random_restarts',
    'strategies',
]
