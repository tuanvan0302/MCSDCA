from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


ParamList = list[nn.Parameter]
TensorList = list[torch.Tensor]


def trainable_parameters(parameters: Iterable[nn.Parameter]) -> ParamList:
    """Return trainable parameters as a stable list."""

    params = [param for param in parameters if param.requires_grad]
    if not params:
        raise ValueError("MCSDCA requires at least one trainable parameter.")
    return params


@torch.no_grad()
def clone_params(parameters: Iterable[nn.Parameter]) -> TensorList:
    """Detach and clone parameters while preserving device and dtype."""

    return [param.detach().clone() for param in parameters]


@torch.no_grad()
def assign_params(parameters: Iterable[nn.Parameter], values: Iterable[torch.Tensor]) -> None:
    """Copy tensor values into parameters."""

    for param, value in zip(parameters, values, strict=True):
        param.copy_(value)


def zero_like(values: Iterable[torch.Tensor]) -> TensorList:
    return [torch.zeros_like(value) for value in values]


def clone_tensors(values: Iterable[torch.Tensor]) -> TensorList:
    return [value.detach().clone() for value in values]


def dca_closed_form_update(
    base: TensorList,
    y: TensorList,
    gamma_k: float,
    local_entropy_time: float,
) -> TensorList:
    """Closed-form update for the local-entropy DCA convex subproblem."""

    alpha = local_entropy_time * gamma_k / (1.0 + local_entropy_time * gamma_k)
    beta = 1.0 / (1.0 + local_entropy_time * gamma_k)
    return [alpha * base_value + beta * y_value for base_value, y_value in zip(base, y, strict=True)]


def resolve_base_gamma(config) -> float:
    """Base proximal coefficient gamma_0 for the DCA outer step.

    When ``config.beta0`` is set it is the target initial DCA mixing weight
    ``beta = 1/(1 + t*gamma_0)`` (fraction of ``y_k`` in ``x_{k+1}``), so
    ``gamma_0 = (1/beta0 - 1) / t``. Otherwise ``config.gamma`` is used directly.
    """

    if getattr(config, "beta0", None) is not None:
        return (1.0 / config.beta0 - 1.0) / config.local_entropy_time
    return config.gamma


def gamma_at_step(gamma: float, gamma_power: float, outer_step: int) -> float:
    return gamma * float(outer_step + 1) ** gamma_power


def markov_chain_length_at_step(base_length: int, length_power: float, outer_step: int) -> int:
    return base_length + math.floor(float(outer_step + 1) ** length_power)


def finite_or_raise(loss: torch.Tensor, name: str = "loss") -> None:
    if loss.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor.")
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"{name} is not finite: {float(loss.detach().cpu())}")


def stable_sqrt(value: float) -> float:
    return math.sqrt(max(value, 0.0))