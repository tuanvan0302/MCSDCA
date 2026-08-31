from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import nn

from .config import MCSDCAConfig
from .param_utils import (
    assign_params,
    clip_grads,
    clone_params,
    clone_tensors,
    dca_closed_form_update,
    finite_or_raise,
    gamma_at_step,
    markov_chain_length_at_step,
    resolve_base_gamma,
    stable_sqrt,
    trainable_parameters,
    zero_like,
)


class MCSDCAUdLD:
    """MCSDCA optimizer using underdamped Langevin dynamics."""

    def __init__(self, parameters: Iterable[nn.Parameter], config: MCSDCAConfig):
        config.validate()
        self.params = trainable_parameters(parameters)
        self.config = config
        self.outer_step = 0
        self.backprop_calls = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self, loss_fn: Callable[[], torch.Tensor]) -> dict[str, float | int]:
        cfg = self.config
        base = clone_params(self.params)
        chain = clone_tensors(base)
        velocity = zero_like(base)
        y_k_sum = zero_like(base)
        sample_count = 0
        loss_sum = 0.0
        transition_count = 0

        delta = cfg.ud_delta
        exp2 = math.exp(-2.0 * delta)
        exp4 = math.exp(-4.0 * delta)
        c1 = cfg.epsilon * (delta - 0.25 * exp4 - 0.75 + exp2)
        c2 = cfg.epsilon * (1.0 - exp4)
        c3 = 0.5 * cfg.epsilon * (1.0 + exp4 - 2.0 * exp2)
        conditional_var = max(c1 - (c3 * c3 / c2 if c2 > 0 else 0.0), 0.0)

        sqrt_c2 = stable_sqrt(c2)
        sqrt_conditional = stable_sqrt(conditional_var)
        drift_x_factor = 0.5 * delta - 0.25 * (1.0 - exp2)
        chain_length = markov_chain_length_at_step(
            cfg.langevin_steps,
            cfg.langevin_steps_power,
            self.outer_step,
        )
        if cfg.max_langevin_steps is not None:
            chain_length = min(chain_length, cfg.max_langevin_steps)

        if cfg.burn_in == 0:
            for total, value in zip(y_k_sum, chain, strict=True):
                total.add_(value)
            sample_count += 1

        for inner_step in range(chain_length - 1):
            assign_params(self.params, chain)
            self.zero_grad()

            loss = loss_fn()
            finite_or_raise(loss)
            loss.backward()

            grads = [param.grad.detach().clone() for param in self.params]
            grads = clip_grads(grads, cfg.max_grad_norm)
            loss_sum += float(loss.detach().cpu())
            transition_count += 1
            self.backprop_calls += 1

            next_velocity: list[torch.Tensor] = []
            next_chain: list[torch.Tensor] = []
            for value, base_value, vel, grad in zip(chain, base, velocity, grads, strict=True):
                target_grad = grad + (value - base_value) / cfg.local_entropy_time
                expected_v = exp2 * vel - 0.5 * (1.0 - exp2) * target_grad
                sampled_v = expected_v + sqrt_c2 * torch.randn_like(vel)

                expected_x = value + 0.5 * (1.0 - exp2) * vel - drift_x_factor * target_grad
                if c2 > 0:
                    correlated_noise = (c3 / c2) * (sampled_v - expected_v)
                else:
                    correlated_noise = torch.zeros_like(value)
                sampled_x = expected_x + correlated_noise + sqrt_conditional * torch.randn_like(value)

                next_velocity.append(sampled_v.detach())
                next_chain.append(sampled_x.detach())

            velocity = next_velocity
            chain = next_chain

            state_index = inner_step + 1
            if state_index >= cfg.burn_in:
                for total, value in zip(y_k_sum, chain, strict=True):
                    total.add_(value)
                sample_count += 1

        if sample_count <= 0:
            raise ValueError("MCSDCA retained no Markov-chain samples. Check burn_in and chain length.")
        y_k = [total / float(sample_count) for total in y_k_sum]
        gamma_k = gamma_at_step(resolve_base_gamma(cfg), cfg.gamma_power, self.outer_step)
        updated = dca_closed_form_update(base, y_k, gamma_k, cfg.local_entropy_time)
        assign_params(self.params, updated)
        self.zero_grad()

        self.outer_step += 1
        return {
            "loss": loss_sum / float(transition_count),
            "outer_step": self.outer_step,
            "backprop_calls": self.backprop_calls,
            "markov_chain_length": chain_length,
            "retained_samples": sample_count,
            "gamma_k": gamma_k,
        }
