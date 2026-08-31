from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCSDCAConfig:
    """Hyperparameters shared by MCSDCA-odLD and MCSDCA-udLD.

    Defaults follow the paper's deep-learning setup (t = 1e4, gamma_power = 0.1,
    delta = 0.1) except for two deliberate departures used in these experiments:
    ``epsilon`` is raised from the paper's 1e-8 (viscosity-vanishing) to 1e-2 for
    genuine Langevin exploration, and ``od_eta`` from 1e-3 to 3e-3 (1e-3 stalls,
    per 00_MCSDCA_paper_experiments). Markov chains are kept short (5/2 vs the
    paper's 20/10) for compute. All are meant to be swept.
    """

    langevin_steps: int = 5  # Base Markov-chain length n_k = base + floor((k+1)^power)
    langevin_steps_power: float = 0.1  # Theorem 1 lambda for growing chain length
    max_langevin_steps: int | None = 8  # Practical cap for long runs
    burn_in: int = 2  # Number of initial Markov-chain states to discard
    local_entropy_time: float = 1e4  # PDE evolution time t
    gamma: float = 1e-5  # Proximal base; ignored when beta0 is set
    gamma_power: float = 0.1  # gamma_k = base_gamma * (k+1)^gamma_power
    beta0: float | None = None  # Initial DCA mix 1/(1+t*gamma_0); overrides gamma when set
    epsilon: float = 1e-2  # Langevin temperature / PDE viscosity
    od_eta: float = 3e-3  # Overdamped Langevin step size
    ud_delta: float = 0.1  # Underdamped Langevin step size
    max_grad_norm: float | None = 1.0

    def validate(self) -> None:
        if self.langevin_steps <= 0:
            raise ValueError("langevin_steps must be positive.")
        if self.langevin_steps_power <= 0:
            raise ValueError("langevin_steps_power must be positive.")
        if self.max_langevin_steps is not None and self.max_langevin_steps < self.langevin_steps:
            raise ValueError("max_langevin_steps must be at least langevin_steps when set.")
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
        if self.beta0 is not None and not 0.0 < self.beta0 < 1.0:
            raise ValueError("beta0 must be in (0, 1) when set.")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if self.od_eta <= 0:
            raise ValueError("od_eta must be positive.")
        if self.ud_delta <= 0:
            raise ValueError("ud_delta must be positive.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when set.")
