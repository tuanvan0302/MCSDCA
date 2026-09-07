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
    paper's 20/10) for compute.

    ``n_k`` and ``gamma_k`` are adaptive during training and NOT swept: chain
    length grows as ``langevin_steps + floor((k+1)^langevin_steps_power)`` and
    the proximal coefficient as ``gamma_0 * (k+1)^gamma_power``.

    The Experiment-1 sweeps vary ``od_eta`` over {3e-3, 1e-2, 3e-2} and
    ``epsilon`` as the RATIO epsilon/eta over {1e-6, 1e-2, 1e0} (see
    src/sweep.py), so the Langevin noise-to-signal regime is the knob rather
    than an eta-dependent absolute; ``ud_delta`` over {0.03, 0.1, 0.3}.

    For a faithful, un-tuned reproduction use :meth:`paper` (long chains, the
    paper's epsilon / step size / gamma schedule) with the ``paper`` training
    profile.
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

    @classmethod
    def paper(cls) -> "MCSDCAConfig":
        """The paper's deep-learning setup, reproduced verbatim (no shortcuts).

        From "Training procedure" (Chaudhari et al. 2019 local-entropy schedule):

          * time  1/t = 1e-4  (t = 1e4); scoping exponent held at k = 1
          * epsilon = 1e-8  (viscosity-vanishing regime)
          * Langevin step size (Eq. 7) = 1e-3  -> od_eta (and ud_delta)
          * chain length  n_k = 20 + floor((k+1)^0.1), NO cap; discard 10 states
          * gamma_k = (1/t) * gamma_tilde_k  with  gamma_tilde_k = 1e-5 (k+1)^0.1,
            hence gamma_0 = 1e-4 * 1e-5 = 1e-9 (stays in sync with t = 1e4 here);
            with beta0 unset this reproduces gamma_k exactly.

        Pairs with the ``paper`` training profile, whose AdamW baseline mirrors
        le-wm/config/train/lewm.yaml (AdamW lr 5e-5 / wd 1e-3, bf16, batch 128,
        grad-clip 1.0, 100 epochs, SIGReg weight 0.09 / knots 17 / num_proj 1024,
        LinearWarmupCosineAnnealingLR).
        """

        return cls(
            langevin_steps=20,
            langevin_steps_power=0.1,
            max_langevin_steps=None,
            burn_in=10,
            local_entropy_time=1e4,
            gamma=1e-9,
            gamma_power=0.1,
            beta0=None,
            epsilon=1e-8,
            od_eta=1e-3,
            ud_delta=1e-3,
            max_grad_norm=1.0,
        )

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
