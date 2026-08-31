"""Fast CPU smoke test for the full-model MCSDCA training path.

Exercises `predict_target`, `select_full_model_parameters`, one MCSDCA-odLD /
MCSDCA-udLD outer step and one baseline step on a tiny LeWM-shaped model, and
checks that every one of the five module groups is updated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import make_baseline_optimizer
from src.lewm_predictor import (
    LEWM_TRAINING_MODULES,
    predict_target,
    select_full_model_parameters,
)
from src.mcsdca import MCSDCAConfig, MCSDCAOdLD, MCSDCAUdLD


class ToyLeWM(nn.Module):
    """Small LeWM-shaped model with all five trainable module groups."""

    def __init__(self, obs_dim: int = 8, action_dim: int = 3, emb_dim: int = 5):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, emb_dim)
        self.projector = nn.Linear(emb_dim, emb_dim)
        self.action_encoder = nn.Linear(action_dim, emb_dim)
        self.predictor = nn.Sequential(nn.Linear(2 * emb_dim, 16), nn.Tanh(), nn.Linear(16, emb_dim))
        self.pred_proj = nn.Linear(emb_dim, emb_dim)

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        emb = self.projector(self.encoder(info["pixels"].float()))
        return {"emb": emb, "act_emb": self.action_encoder(info["action"].float())}

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.pred_proj(self.predictor(torch.cat([emb, act_emb], dim=-1)))


def make_batch(batch_size: int = 4, seq_len: int = 6, obs_dim: int = 8, action_dim: int = 3) -> dict[str, torch.Tensor]:
    action = torch.randn(batch_size, seq_len, action_dim)
    action[0, 0, 0] = float("nan")  # sequence-boundary NaN, must be sanitized
    return {"pixels": torch.randn(batch_size, seq_len, obs_dim), "action": action}


@torch.no_grad()
def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_every_module_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> None:
    for module in LEWM_TRAINING_MODULES:
        changed = any(
            name.startswith(module + ".") and not torch.allclose(value, after[name])
            for name, value in before.items()
        )
        if not changed:
            raise AssertionError(f"Expected parameters under '{module}' to change.")


def loss_closure(model: nn.Module, batch: dict[str, torch.Tensor]):
    def _loss() -> torch.Tensor:
        pred, target, _ = predict_target(model, batch, history_size=3, num_preds=1)
        return F.mse_loss(pred, target)

    return _loss


def check_predict_target() -> None:
    torch.manual_seed(0)
    model = ToyLeWM()
    pred, target, emb = predict_target(model, make_batch(), history_size=3, num_preds=1)
    if pred.shape != target.shape or pred.shape[1] != 3:
        raise AssertionError(f"unexpected shapes: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if emb.shape[1] != 6:
        raise AssertionError("emb should keep the full sequence length for SIGReg reuse.")
    if not torch.isfinite(F.mse_loss(pred, target)):
        raise AssertionError("prediction loss is not finite.")


def check_mcsdca(optimizer_cls, **cfg_kwargs) -> None:
    torch.manual_seed(1)
    model = ToyLeWM()
    batch = make_batch()
    before = clone_state(model)
    optimizer = optimizer_cls(
        select_full_model_parameters(model),
        MCSDCAConfig(langevin_steps=3, langevin_steps_power=1.0, max_langevin_steps=None, burn_in=1, **cfg_kwargs),
    )
    info = optimizer.step(loss_closure(model, batch))
    if info["backprop_calls"] != 3 or info["markov_chain_length"] != 4 or info["retained_samples"] != 3:
        raise AssertionError(f"unexpected chain accounting: {info}")
    info = optimizer.step(loss_closure(model, batch))
    if info["backprop_calls"] != 7 or info["markov_chain_length"] != 5:
        raise AssertionError(f"chain length should grow over outer steps: {info}")
    assert_every_module_changed(before, clone_state(model))


def check_beta0_resolution() -> None:
    from src.mcsdca.param_utils import resolve_base_gamma

    cfg = MCSDCAConfig(beta0=0.9, local_entropy_time=1e4)
    gamma0 = resolve_base_gamma(cfg)
    if not abs(gamma0 - (1.0 / 0.9 - 1.0) / 1e4) < 1e-12:
        raise AssertionError(f"resolve_base_gamma wrong: {gamma0}")
    if abs(resolve_base_gamma(MCSDCAConfig()) - MCSDCAConfig().gamma) > 0:
        raise AssertionError("resolve_base_gamma should fall back to cfg.gamma when beta0 is None.")


def check_baselines() -> None:
    batch = make_batch()
    for name in ("AdamW", "Adam", "SGD + momentum", "RMSprop", "Adagrad"):
        torch.manual_seed(3)
        model = ToyLeWM()
        params = select_full_model_parameters(model)
        before = clone_state(model)
        optimizer = make_baseline_optimizer(name, params, lr=1e-2, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss_closure(model, batch)().backward()
        optimizer.step()
        assert_every_module_changed(before, clone_state(model))


def check_invalid_config() -> None:
    for kwargs in ({"langevin_steps": 2, "burn_in": 3}, {"gamma_power": 1.0}, {"beta0": 1.5}):
        try:
            MCSDCAConfig(**kwargs).validate()
        except ValueError:
            continue
        raise AssertionError(f"MCSDCAConfig({kwargs}) should have raised ValueError.")


def main() -> None:
    check_predict_target()
    check_mcsdca(MCSDCAOdLD, od_eta=1e-2, epsilon=1e-8)
    check_mcsdca(MCSDCAUdLD, ud_delta=0.1, epsilon=1e-8)
    check_beta0_resolution()
    check_baselines()
    check_invalid_config()
    print("Full-model MCSDCA smoke test passed.")


if __name__ == "__main__":
    main()
