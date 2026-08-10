from __future__ import annotations

import copy
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import make_baseline_optimizer
from src.lewm_predictor import (
    freeze_encoder_side,
    latent_rollout_stats,
    one_step_predictor_loss,
    rollout_predictor_loss,
    select_dynamics_head_parameters,
)
from src.mcsdca import MCSDCAConfig, MCSDCAOdLD, MCSDCAUdLD


class ToyLeWM(nn.Module):
    """Small LeWM-like model used only for smoke testing."""

    def __init__(self, obs_dim: int = 8, action_dim: int = 3, emb_dim: int = 5):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, emb_dim)
        self.projector = nn.Linear(emb_dim, emb_dim)
        self.action_encoder = nn.Linear(action_dim, emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(2 * emb_dim, 16),
            nn.Tanh(),
            nn.Linear(16, emb_dim),
        )
        self.pred_proj = nn.Linear(emb_dim, emb_dim)

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = info["pixels"].float()
        action = info["action"].float()
        emb = self.projector(self.encoder(pixels))
        act_emb = self.action_encoder(action)
        return {"emb": emb, "act_emb": act_emb}

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        pred_in = torch.cat([emb, act_emb], dim=-1)
        pred = self.predictor(pred_in)
        return self.pred_proj(pred)


def make_batch(batch_size: int = 4, seq_len: int = 6, obs_dim: int = 8, action_dim: int = 3) -> dict[str, torch.Tensor]:
    pixels = torch.randn(batch_size, seq_len, obs_dim)
    action = torch.randn(batch_size, seq_len, action_dim)
    action[0, 0, 0] = float("nan")
    return {"pixels": pixels, "action": action}


@torch.no_grad()
def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_any_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], prefix: str) -> None:
    changed = [
        name
        for name, old_value in before.items()
        if name.startswith(prefix) and not torch.allclose(old_value, after[name])
    ]
    if not changed:
        raise AssertionError(f"Expected parameters under '{prefix}' to change.")


def assert_all_same(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> None:
    for name, old_value in before.items():
        if name.startswith(prefixes) and not torch.allclose(old_value, after[name]):
            raise AssertionError(f"Expected frozen parameter '{name}' to stay unchanged.")


def check_losses() -> None:
    torch.manual_seed(0)
    model = ToyLeWM()
    batch = make_batch()

    one_step = one_step_predictor_loss(model, batch, history_size=3, num_preds=1)
    rollout = rollout_predictor_loss(model, batch, history_size=3, horizon=2, discount=0.8)
    stats = latent_rollout_stats(model, batch, history_size=3, horizon=2)

    if one_step.ndim != 0 or not torch.isfinite(one_step):
        raise AssertionError("one_step_predictor_loss must return a finite scalar.")
    if rollout.ndim != 0 or not torch.isfinite(rollout):
        raise AssertionError("rollout_predictor_loss must return a finite scalar.")
    if not all(torch.isfinite(torch.tensor(value)) for value in stats.values()):
        raise AssertionError("latent_rollout_stats must return finite values.")


def check_mcsdca_odld() -> None:
    torch.manual_seed(1)
    model = ToyLeWM()
    freeze_encoder_side(model)
    batch = make_batch()
    params = select_dynamics_head_parameters(model)
    before = clone_state(model)

    optimizer = MCSDCAOdLD(
        params,
        MCSDCAConfig(langevin_steps=3, langevin_steps_power=1.0, burn_in=1, od_eta=1e-2, epsilon=1e-8),
    )
    info = optimizer.step(lambda: one_step_predictor_loss(model, batch, history_size=3, num_preds=1))

    if info["backprop_calls"] != 3:
        raise AssertionError("MCSDCA-odLD should report three backprop calls.")
    if info["markov_chain_length"] != 4 or info["retained_samples"] != 3:
        raise AssertionError("MCSDCA-odLD should use scheduled Markov-chain length.")

    info = optimizer.step(lambda: one_step_predictor_loss(model, batch, history_size=3, num_preds=1))
    after = clone_state(model)

    if info["backprop_calls"] != 7:
        raise AssertionError("MCSDCA-odLD should accumulate scheduled backprop calls.")
    if info["markov_chain_length"] != 5 or info["retained_samples"] != 4:
        raise AssertionError("MCSDCA-odLD should grow Markov-chain length over time.")
    assert_any_changed(before, after, "predictor")
    assert_all_same(before, after, ("encoder", "projector"))


def check_mcsdca_udld() -> None:
    torch.manual_seed(2)
    model = ToyLeWM()
    freeze_encoder_side(model)
    batch = make_batch()
    params = select_dynamics_head_parameters(model)
    before = clone_state(model)

    optimizer = MCSDCAUdLD(
        params,
        MCSDCAConfig(langevin_steps=3, langevin_steps_power=1.0, burn_in=1, ud_delta=0.1, epsilon=1e-8),
    )
    info = optimizer.step(lambda: one_step_predictor_loss(model, batch, history_size=3, num_preds=1))

    if info["backprop_calls"] != 3:
        raise AssertionError("MCSDCA-udLD should report three backprop calls.")
    if info["markov_chain_length"] != 4 or info["retained_samples"] != 3:
        raise AssertionError("MCSDCA-udLD should use scheduled Markov-chain length.")

    info = optimizer.step(lambda: one_step_predictor_loss(model, batch, history_size=3, num_preds=1))
    after = clone_state(model)

    if info["backprop_calls"] != 7:
        raise AssertionError("MCSDCA-udLD should accumulate scheduled backprop calls.")
    if info["markov_chain_length"] != 5 or info["retained_samples"] != 4:
        raise AssertionError("MCSDCA-udLD should grow Markov-chain length over time.")
    assert_any_changed(before, after, "predictor")
    assert_all_same(before, after, ("encoder", "projector"))


def check_baselines() -> None:
    batch = make_batch()
    for name in ("AdamW", "Adam", "SGD + momentum", "RMSprop", "Adagrad"):
        torch.manual_seed(3)
        model = ToyLeWM()
        freeze_encoder_side(model)
        params = select_dynamics_head_parameters(model)
        before = clone_state(model)
        optimizer = make_baseline_optimizer(name, params, lr=1e-3, weight_decay=0.0)

        optimizer.zero_grad(set_to_none=True)
        loss = one_step_predictor_loss(model, batch, history_size=3, num_preds=1)
        loss.backward()
        optimizer.step()

        after = clone_state(model)
        assert_any_changed(before, after, "predictor")
        assert_all_same(before, after, ("encoder", "projector"))


def check_invalid_config() -> None:
    try:
        MCSDCAConfig(langevin_steps=2, burn_in=3).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid MCSDCAConfig should raise ValueError.")

    try:
        MCSDCAConfig(gamma_power=1.0).validate()
    except ValueError:
        return
    raise AssertionError("Invalid MCSDCAConfig should raise ValueError.")


def main() -> None:
    check_losses()
    check_mcsdca_odld()
    check_mcsdca_udld()
    check_baselines()
    check_invalid_config()
    print("Predictor MCSDCA smoke test passed.")


if __name__ == "__main__":
    main()
