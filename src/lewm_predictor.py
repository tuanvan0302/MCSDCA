from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _module_parameters(module: nn.Module | None) -> list[nn.Parameter]:
    if module is None:
        return []
    return [param for param in module.parameters() if param.requires_grad]


def select_dynamics_head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Select the predictor-side parameters for experiment 1.

    In LeWM, `predict()` uses action_encoder -> predictor -> pred_proj. The
    encoder/projector side creates latent targets and is kept fixed in this
    first experiment.
    """

    params: list[nn.Parameter] = []
    for name in ("action_encoder", "predictor", "pred_proj"):
        params.extend(_module_parameters(getattr(model, name, None)))
    if not params:
        raise ValueError("No trainable dynamics-head parameters found.")
    return params


def select_core_predictor_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = _module_parameters(getattr(model, "predictor", None))
    if not params:
        raise ValueError("No trainable core predictor parameters found.")
    return params


def freeze_modules(modules: Iterable[nn.Module | None]) -> None:
    for module in modules:
        if module is None:
            continue
        module.requires_grad_(False)


def freeze_encoder_side(model: nn.Module) -> None:
    """Freeze the modules that produce target latents in predictor-only tests."""

    freeze_modules([getattr(model, "encoder", None), getattr(model, "projector", None)])


def sanitize_action(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mirror LeWM training behavior at sequence boundaries."""

    if "action" in batch:
        batch = dict(batch)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    return batch


def encode_latents(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return observation and action embeddings from a LeWM-like model."""

    output = model.encode(sanitize_action(batch))
    if "emb" not in output:
        raise KeyError("model.encode(batch) must return key 'emb'.")
    if "act_emb" not in output:
        raise KeyError("model.encode(batch) must return key 'act_emb'.")
    return output["emb"], output["act_emb"]


def one_step_predictor_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    history_size: int,
    num_preds: int = 1,
    detach_targets: bool = True,
) -> torch.Tensor:
    """Compute the one-step latent prediction loss used by LeWM training."""

    emb, act_emb = encode_latents(model, batch)
    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    target = emb[:, num_preds : num_preds + history_size]
    if target.size(1) != ctx_emb.size(1):
        raise ValueError(
            f"Prediction/target length mismatch: {ctx_emb.size(1)} != {target.size(1)}."
        )
    if detach_targets:
        target = target.detach()
    pred = model.predict(ctx_emb, ctx_act)
    return F.mse_loss(pred, target)


def rollout_predictor_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    history_size: int,
    horizon: int,
    discount: float = 1.0,
    detach_targets: bool = True,
) -> torch.Tensor:
    """Autoregressively rollout predictor and compare to future latents.

    The initial history is encoded from observations. Each future latent is
    produced from the latest `history_size` predicted/observed latents and the
    corresponding action embeddings.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if discount <= 0:
        raise ValueError("discount must be positive.")

    emb, act_emb = encode_latents(model, batch)
    if emb.size(1) < history_size + horizon:
        raise ValueError("Batch sequence length is too short for requested rollout horizon.")

    latents = emb[:, :history_size]
    losses: list[torch.Tensor] = []

    for step in range(horizon):
        act_window = act_emb[:, step : step + history_size]
        pred_next = model.predict(latents[:, -history_size:], act_window)[:, -1:]
        target = emb[:, history_size + step : history_size + step + 1]
        if detach_targets:
            target = target.detach()
        weight = discount**step
        losses.append(weight * F.mse_loss(pred_next, target))
        latents = torch.cat([latents, pred_next], dim=1)

    return torch.stack(losses).sum() / float(len(losses))


@torch.no_grad()
def latent_rollout_stats(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    history_size: int,
    horizon: int,
) -> dict[str, float]:
    """Compute simple rollout diagnostics without updating parameters."""

    emb, act_emb = encode_latents(model, batch)
    latents = emb[:, :history_size]
    predicted: list[torch.Tensor] = []

    for step in range(horizon):
        act_window = act_emb[:, step : step + history_size]
        pred_next = model.predict(latents[:, -history_size:], act_window)[:, -1:]
        predicted.append(pred_next)
        latents = torch.cat([latents, pred_next], dim=1)

    pred = torch.cat(predicted, dim=1)
    target = emb[:, history_size : history_size + horizon]
    drift = pred.norm(dim=-1).mean() - target.norm(dim=-1).mean()
    return {
        "rollout_mse": float(F.mse_loss(pred, target).detach().cpu()),
        "pred_latent_norm": float(pred.norm(dim=-1).mean().detach().cpu()),
        "target_latent_norm": float(target.norm(dim=-1).mean().detach().cpu()),
        "latent_norm_drift": float(drift.detach().cpu()),
        "pred_latent_variance": float(pred.var(dim=(0, 1)).mean().detach().cpu()),
    }

