from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCSDCAConfig:
    """Hyperparameters shared by MCSDCA-odLD and MCSDCA-udLD.

    Defaults reproduce the paper's deep-learning "Training procedure" verbatim
    (Chaudhari et al. 2019 local-entropy schedule):

      * time  1/t = 1e-4  (t = 1e4); scoping held fixed at k = 1 (standard setup;
        the paper's advanced 1/t = 1e-4 * 1.001^k scoping is not used here)
      * epsilon = 1e-8  (viscosity-vanishing regime)
      * Langevin step size (Algorithm 2, Eq. 7) = 1e-3  -> eta (odLD only)
      * underdamped discretization coefficient (Algorithm 3) = 0.1  -> delta (udLD only)
      * chain length  n_k = n_init + floor(k * lambda) = 20 + floor(k * 0.1), NO cap;
        discard 10 states
      * gamma_k = (1/t) * gamma_tilde_k  with  gamma_tilde_k = 1e-5 (k+1)^0.1,
        hence gamma_0 = 1e-4 * 1e-5 = 1e-9 (in sync with t = 1e4); beta0 unset
        reproduces gamma_k exactly.

    ``n_k`` and ``gamma_k`` are adaptive during training: the chain length grows
    as ``langevin_steps + floor((k+1) * langevin_steps_rate)`` and the proximal
    coefficient as ``gamma_0 * (k+1)^gamma_power``.

    The paper's Langevin inner loop uses the raw stochastic gradient -- there is
    deliberately NO per-step gradient-norm clipping here (the AdamW baseline keeps
    its own ``gradient_clip_val`` from le-wm/config/train/lewm.yaml).
    """

    langevin_steps: int = 20  # Base Markov-chain length n_init; n_k = n_init + floor((k+1) * rate)
    langevin_steps_rate: float = 0.1  # Theorem 1 lambda: linear growth rate of chain length
    max_langevin_steps: int | None = None  # Practical cap for long runs (None = no cap, per paper)
    burn_in: int = 10  # Number of initial Markov-chain states to discard
    local_entropy_time: float = 1e4  # PDE evolution time t
    gamma: float = 1e-9  # Proximal base gamma_0; ignored when beta0 is set
    gamma_power: float = 0.1  # gamma_k = base_gamma * (k+1)^gamma_power
    beta0: float | None = None  # Initial DCA mix 1/(1+t*gamma_0); overrides gamma when set
    epsilon: float = 1e-8  # Langevin temperature / PDE viscosity
    eta: float = 1e-3  # Overdamped Langevin step size (Algorithm 2, odLD only)
    delta: float = 0.1  # Underdamped Langevin discretization coefficient (Algorithm 3, udLD only)

    @classmethod
    def paper(cls) -> "MCSDCAConfig":
        """Alias for the default constructor -- the defaults already are the
        paper's setup. Kept for readability at call sites and reproduction tests.
        """

        return cls()

    def validate(self) -> None:
        if self.langevin_steps <= 0:
            raise ValueError("langevin_steps must be positive.")
        if self.langevin_steps_rate <= 0:
            raise ValueError("langevin_steps_rate must be positive.")
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
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        if self.delta <= 0:
            raise ValueError("delta must be positive.")
