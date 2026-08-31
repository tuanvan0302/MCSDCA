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
    trainable_parameters,
    zero_like,
)


class MCSDCAOdLD:
    """MCSDCA optimizer using overdamped Langevin dynamics."""

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
        """Run one MCSDCA outer iteration.

        The closure must compute a scalar loss from the current parameter
        values. Parameters are temporarily moved along the Markov chain during
        sampling, then replaced by the DCA closed-form update.
        """

        cfg = self.config
        base = clone_params(self.params)
        chain = clone_tensors(base)
        y_k_sum = zero_like(base)
        sample_count = 0
        loss_sum = 0.0
        transition_count = 0
        noise_scale = math.sqrt(2.0 * cfg.od_eta * cfg.epsilon)
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

            next_chain = []
            for value, base_value, grad in zip(chain, base, grads, strict=True):
                target_grad = grad + (value - base_value) / cfg.local_entropy_time
                noise = noise_scale * torch.randn_like(value)
                next_chain.append((value - cfg.od_eta * target_grad + noise).detach())
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
