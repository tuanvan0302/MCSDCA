from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def make_baseline_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create baseline optimizers for comparison with MCSDCA."""

    params = [param for param in parameters if param.requires_grad]
    if not params:
        raise ValueError("Baseline optimizer requires at least one trainable parameter.")

    key = "".join(char for char in name.lower() if char.isalnum())
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key in {"sgd", "sgdmomentum"}:
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if key == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if key == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    supported = "AdamW, Adam, SGD + momentum, RMSprop, Adagrad"
    raise ValueError(f"Unsupported optimizer '{name}'. Supported: {supported}.")
