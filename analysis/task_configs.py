"""
Shared task configuration for analysis notebooks.

Each results / diagnostics / hp_sensitivity notebook selects a task with::

    from task_configs import TASK_CONFIGS
    cfg = TASK_CONFIGS["mnist_mlp"]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "parameters"))

from optimizer_registry import ALL_OPTIMIZERS  # noqa: E402


TASK_CONFIGS = {
    "mnist_mlp": {
        "task_tag": "mnist_mlp",
        "display_name": "MNIST MLP",
        "iteration": 4,
        "optimizers": ALL_OPTIMIZERS,
        "sort_metric": "final_max_val_acc",
        "sort_order": "-",
        "metric_key": "sweep_metric",
        "direction": "minimize",
        "speedrun_targets": [0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.975],
        "speedrun_direction": "above",
        "speedrun_metric": "test_acc",
        "history_metrics": ["train_loss", "train_acc", "test_acc", "train_time_seconds"],
        "hist_x": "final_max_acc_epoch",
        "hist_y": "final_max_val_acc",
        "best_epoch_key": "final_max_acc_epoch",
        "best_metric_key": "test_acc",
        "highlight": ["sgd_learn_diag_curv", "sgd_learn_diag_curv_log"],
        "y_keys": ["train_loss", "test_error"],
        "y_labels": {"train_loss": "Train Loss", "test_error": "Test Error (1 - acc)"},
        "y_transforms": {"test_error": lambda d: 1.0 - d["test_acc"]},
        "log_y": True,
        "convergence_threshold": -0.9,
    },
    "cifar10_resnet18": {
        "task_tag": "cifar10_resnet18",
        "display_name": "CIFAR-10 ResNet-18",
        "iteration": 0,
        "optimizers": ["adam", "adamw", "sgd", "sgd_metric", "sgd_log_metric", "muon", "sgd_rms"],
        "sort_metric": "final_max_val_acc",
        "sort_order": "-",
        "metric_key": "sweep_metric",
        "direction": "minimize",
        "speedrun_targets": [0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88],
        "speedrun_direction": "above",
        "speedrun_metric": "test_acc",
        "history_metrics": ["train_loss", "train_acc", "test_acc", "train_time_seconds"],
        "hist_x": "final_max_acc_epoch",
        "hist_y": "final_max_val_acc",
        "best_epoch_key": "final_max_acc_epoch",
        "best_metric_key": "test_acc",
        "highlight": ["sgd_learn_diag_curv", "sgd_learn_diag_curv_log"],
        "y_keys": ["train_loss", "test_error"],
        "y_labels": {"train_loss": "Train Loss", "test_error": "Test Error (1 - acc)"},
        "y_transforms": {"test_error": lambda d: 1.0 - d["test_acc"]},
        "log_y": True,
        "convergence_threshold": -0.7,
    },
    "regression": {
        "task_tag": "regression",
        "display_name": "Regression",
        "iteration": 0,
        "optimizers": ["adam", "adamw", "sgd", "sgd_metric", "sgd_log_metric", "muon", "sgd_rms"],
        "sort_metric": "final_min_val_loss",
        "sort_order": "+",
        "metric_key": "sweep_metric",
        "direction": "minimize",
        "speedrun_targets": [0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        "speedrun_direction": "below",
        "speedrun_metric": "test_mse",
        "history_metrics": ["train_loss", "train_mse", "test_mse", "train_time_seconds"],
        "hist_x": "final_min_loss_epoch",
        "hist_y": "final_min_val_loss",
        "best_epoch_key": "final_min_loss_epoch",
        "best_metric_key": "test_mse",
        "highlight": ["sgd_learn_diag_curv", "sgd_learn_diag_curv_log"],
        "y_keys": ["train_loss", "test_mse"],
        "y_labels": {"train_loss": "Train Loss", "test_mse": "Test MSE"},
        "y_transforms": {},
        "log_y": True,
        "convergence_threshold": None,
    },
    "shakespeare_minigpt": {
        "task_tag": "shakespeare_minigpt",
        "display_name": "Shakespeare MiniGPT",
        "iteration": 2,
        "optimizers": ALL_OPTIMIZERS,
        "sort_metric": "final_min_val_perplexity",
        "sort_order": "+",
        "metric_key": "sweep_metric",
        "direction": "minimize",
        "speedrun_targets": [12.0, 10.0, 8.0, 6.0, 5.0, 4.6, 4.5, 4.4, 4.0, 3.5],
        "speedrun_direction": "below",
        "speedrun_metric": "val_perplexity",
        "history_metrics": ["train_loss", "val_perplexity", "train_time_seconds"],
        "hist_x": "final_min_perp_epoch",
        "hist_y": "final_min_val_perplexity",
        "best_epoch_key": "final_min_perp_epoch",
        "best_metric_key": "val_perplexity",
        "highlight": ["sgd_learn_diag_curv", "sgd_learn_diag_curv_log"],
        "y_keys": ["train_loss", "val_perplexity"],
        "y_labels": {"train_loss": "Train Loss", "val_perplexity": "Val Perplexity"},
        "y_transforms": {},
        "log_y": True,
        "ylim": {"train_loss": (None, 30), "val_perplexity": (4, 40)},
        "convergence_threshold": None,
    },
}
