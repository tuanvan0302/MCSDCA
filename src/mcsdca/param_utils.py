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


def gamma_at_step(gamma: float, gamma_power: float, outer_step: int) -> float:
    return gamma * float(outer_step + 1) ** gamma_power


def markov_chain_length_at_step(base_length: int, length_power: float, outer_step: int) -> int:
    return base_length + math.floor(float(outer_step + 1) ** length_power)


def finite_or_raise(loss: torch.Tensor, name: str = "loss") -> None:
    if loss.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor.")
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"{name} is not finite: {float(loss.detach().cpu())}")


def clip_grads(grads: TensorList, max_norm: float | None) -> TensorList:
    if max_norm is None:
        return grads

    total_sq = torch.zeros((), device=grads[0].device)
    for grad in grads:
        total_sq = total_sq + grad.detach().pow(2).sum()
    total_norm = torch.sqrt(total_sq)

    if total_norm <= max_norm:
        return grads

    scale = max_norm / (float(total_norm.detach().cpu()) + 1e-12)
    return [grad * scale for grad in grads]


def noise_like(values: TensorList, scale: float) -> TensorList:
    if scale == 0.0:
        return zero_like(values)
    return [scale * torch.randn_like(value) for value in values]


def assert_same_shapes(left: TensorList, right: TensorList) -> None:
    if len(left) != len(right):
        raise ValueError("Parameter lists have different lengths.")
    for lhs, rhs in zip(left, right, strict=True):
        if lhs.shape != rhs.shape:
            raise ValueError(f"Shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}")


def stable_sqrt(value: float) -> float:
    return math.sqrt(max(value, 0.0))