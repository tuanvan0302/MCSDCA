from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCSDCAConfig:
    """Hyperparameters shared by MCSDCA-odLD and MCSDCA-udLD."""

    langevin_steps: int = 4 # Base Markov-chain length in n_k = base + floor(k^lambda)
    langevin_steps_power: float = 0.5 # Theorem 1 lambda for growing Markov-chain length
    burn_in: int = 1 # Number of initial Markov-chain states to discard
    local_entropy_time: float = 1e4
    gamma: float = 1e-5
    gamma_power: float = 0.1
    epsilon: float = 1e-8
    od_eta: float = 1e-3
    ud_delta: float = 0.1
    max_grad_norm: float | None = None

    def validate(self) -> None:
        if self.langevin_steps <= 0:
            raise ValueError("langevin_steps must be positive.")
        if self.langevin_steps_power <= 0:
            raise ValueError("langevin_steps_power must be positive.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if self.burn_in > self.langevin_steps:
            raise ValueError("burn_in must be no larger than langevin_steps.")
        if self.local_entropy_time <= 0:
            raise ValueError("local_entropy_time must be positive.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if not 0 <= self.gamma_power < 1:
            raise ValueError("gamma_power must be in [0, 1).")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if self.od_eta <= 0:
            raise ValueError("od_eta must be positive.")
        if self.ud_delta <= 0:
            raise ValueError("ud_delta must be positive.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when set.")