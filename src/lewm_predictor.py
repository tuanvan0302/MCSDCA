from __future__ import annotations

import torch

LEWM_TRAINING_MODULES = ("encoder", "projector", "action_encoder", "predictor", "pred_proj")


def sanitize_action(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace NaN actions (sequence boundaries) with 0, mirroring LeWM training."""

    if "action" in batch:
        batch = dict(batch)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    return batch


def encode_latents(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (obs_emb, act_emb) from a LeWM-like model's ``encode()``."""

    output = model.encode(sanitize_action(batch))
    if "emb" not in output or "act_emb" not in output:
        raise KeyError("model.encode(batch) must return keys 'emb' and 'act_emb'.")
    return output["emb"], output["act_emb"]


def predict_target(
    model,
    batch: dict[str, torch.Tensor],
    history_size: int,
    num_preds: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a LeWM batch and return ``(prediction, target, emb)``.

    Mirrors ``le-wm/train.py::lejepa_forward``: context = first ``history_size``
    frames, target = the same window shifted forward by ``num_preds``
    (teacher-forced next-step prediction). ``emb`` is returned so callers can
    reuse it (e.g. for SIGReg) without re-encoding.
    """

    emb, act_emb = encode_latents(model, batch)
    context = emb[:, :history_size]
    target = emb[:, num_preds : num_preds + history_size]
    prediction = model.predict(context, act_emb[:, :history_size])
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    return prediction, target, emb


def select_full_model_parameters(model) -> list[torch.nn.Parameter]:
    """Enable grads on the five LeWM modules and return their unique parameters."""

    params: list[torch.nn.Parameter] = []
    missing: list[str] = []
    for name in LEWM_TRAINING_MODULES:
        module = getattr(model, name, None)
        if module is None:
            missing.append(name)
            continue
        module.requires_grad_(True)
        params.extend(module.parameters())
    if missing:
        raise ValueError(f"LeWM model is missing required training modules: {', '.join(missing)}.")
    unique_params = list(dict.fromkeys(params))
    if not unique_params:
        raise ValueError("No trainable full-model parameters found.")
    return unique_params


def enable_full_model_training(model) -> None:
    """Enable gradients on all five LeWM modules and switch to train mode."""

    select_full_model_parameters(model)
    model.train()
