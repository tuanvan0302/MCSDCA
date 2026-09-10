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

    # Fused foreach kernels: one launch for the whole param step instead of a
    # Python loop. Stable for Adam/AdamW on CUDA and a clear win when the model
    # is small (kernel-launch bound), which is the case here.
    fused = {"fused": True} if params[0].is_cuda else {}

    key = "".join(char for char in name.lower() if char.isalnum())
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **fused)
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **fused)

    raise ValueError(f"Unsupported baseline optimizer '{name}'. Supported: AdamW, Adam.")
