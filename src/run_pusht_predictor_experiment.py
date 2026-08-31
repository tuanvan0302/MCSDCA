"""Train and evaluate MCSDCA-odLD / MCSDCA-udLD against baseline optimizers as a
full-model LeWM PushT predictor optimizer.

All five LeWM modules (encoder, projector, action_encoder, predictor, pred_proj)
are initialized from scratch and trained jointly on
``prediction MSE + sigreg_weight * SIGReg`` (the LeWM objective). Every optimizer
gets the same initial weights, the same batch stream, and the same *backprop
budget* (number of backward passes), so results are compared on the
``backprop_calls`` axis. Planning/CEM evaluation lives in ``evaluate_planning.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multiprocess_compat import patch_multiprocess_resource_tracker

patch_multiprocess_resource_tracker()

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.baselines import make_baseline_optimizer
from src.lewm_predictor import (
    encode_latents,
    enable_full_model_training,
    predict_target,
    select_full_model_parameters,
)
from src.mcsdca import MCSDCAConfig, MCSDCAOdLD, MCSDCAUdLD

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
ROLLOUT_HORIZONS = (1, 3, 5)
BASELINE_OPTIMIZERS = ("AdamW", "Adam", "SGD + momentum", "RMSprop", "Adagrad")
MCSDCA_OPTIMIZERS = ("MCSDCA-odLD", "MCSDCA-udLD")
DEFAULT_OPTIMIZERS = ("AdamW", "MCSDCA-odLD", "MCSDCA-udLD")


# --------------------------------------------------------------------------- #
# Training profiles                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingProfile:
    data_fraction: float
    epochs: int  # only used to derive the default backprop budget
    batch_size: int
    eval_batch_size: int
    eval_train_batches: int
    val_batches: int
    precision: str
    sigreg_num_proj: int
    mcsdca: MCSDCAConfig
    history_size: int = 3
    num_preds: int = 1
    frameskip: int = 5
    gradient_clip_val: float = 1.0
    train_fraction: float = 0.9  # episode-level train/val split
    val_eval_fraction: float = 0.15  # fraction of val windows kept for evaluation
    default_budget: int | None = None  # overrides steps_per_epoch * epochs when set


TRAINING_PROFILES: dict[str, TrainingProfile] = {
    "smoke": TrainingProfile(
        data_fraction=0.02,
        epochs=1,
        batch_size=8,
        eval_batch_size=4,
        eval_train_batches=2,
        val_batches=2,
        precision="fp32",
        sigreg_num_proj=64,
        default_budget=40,
        mcsdca=MCSDCAConfig(langevin_steps=3, langevin_steps_power=0.25, max_langevin_steps=4, burn_in=1),
    ),
    "small": TrainingProfile(
        data_fraction=0.10,
        epochs=10,
        batch_size=32,
        eval_batch_size=4,
        eval_train_batches=8,
        val_batches=8,
        precision="fp32",
        sigreg_num_proj=256,
        mcsdca=MCSDCAConfig(),
    ),
    "medium": TrainingProfile(
        data_fraction=0.50,
        epochs=30,
        batch_size=64,
        eval_batch_size=4,
        eval_train_batches=16,
        val_batches=16,
        precision="bf16",
        sigreg_num_proj=512,
        mcsdca=MCSDCAConfig(),
    ),
    "full": TrainingProfile(
        data_fraction=1.0,
        epochs=100,
        batch_size=128,
        eval_batch_size=8,
        eval_train_batches=16,
        val_batches=16,
        precision="bf16",
        sigreg_num_proj=1024,
        mcsdca=MCSDCAConfig(),
    ),
}


def resolve_profile(args: argparse.Namespace) -> TrainingProfile:
    base = TRAINING_PROFILES[args.training_profile]
    overrides: dict[str, Any] = {}
    if args.data_fraction is not None:
        overrides["data_fraction"] = args.data_fraction
    if args.precision is not None:
        overrides["precision"] = args.precision
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.sigreg_num_proj is not None:
        overrides["sigreg_num_proj"] = args.sigreg_num_proj
    return replace(base, **overrides) if overrides else base


def resolve_mcsdca_config(base: MCSDCAConfig, args: argparse.Namespace) -> MCSDCAConfig:
    overrides: dict[str, Any] = {}
    for attr, key in (
        ("mcsdca_epsilon", "epsilon"),
        ("mcsdca_eta", "od_eta"),
        ("mcsdca_delta", "ud_delta"),
        ("mcsdca_beta0", "beta0"),
        ("mcsdca_langevin_steps", "langevin_steps"),
        ("mcsdca_burn_in", "burn_in"),
    ):
        value = getattr(args, attr)
        if value is not None:
            overrides[key] = value
    cfg = replace(base, **overrides) if overrides else base
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# Model + IO helpers                                                           #
# --------------------------------------------------------------------------- #
def require_hdf5() -> Any:
    try:
        import hdf5plugin  # noqa: F401
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "PushT HDF5 pixels use Blosc compression. Install: uv pip install hdf5plugin h5py"
        ) from exc
    return h5py


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optimizer_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def parse_optimizers(value: str) -> list[str]:
    names = [*BASELINE_OPTIMIZERS, *MCSDCA_OPTIMIZERS]
    if value.strip().lower() == "all":
        return list(names)
    aliases = {optimizer_key(name): name for name in names}
    selected: list[str] = []
    for raw in value.split(","):
        key = optimizer_key(raw.strip())
        if key not in aliases:
            raise ValueError(f"Unsupported optimizer '{raw}'. Options: all, {', '.join(names)}")
        selected.append(aliases[key])
    return selected


def initialize_lewm(config_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing LeWM model config: {config_path}")
    config = read_json(config_path)
    model = instantiate(OmegaConf.create(config)).to(device)
    enable_full_model_training(model)
    return model, config


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_state(model: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()}, strict=True)
    enable_full_model_training(model)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# PushT HDF5 sampler (deterministic sequence windows only)                     #
# --------------------------------------------------------------------------- #
class PushTHDF5Sampler:
    def __init__(
        self,
        path: Path,
        frameskip: int,
        max_seq_len: int,
        action_stats_samples: int,
        seed: int,
        train_fraction: float,
    ) -> None:
        self.h5py = require_hdf5()
        self.path = path
        self.frameskip = frameskip
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.raw_span = frameskip * max_seq_len

        with self.h5py.File(path, "r") as h5:
            self.ep_len = np.asarray(h5["ep_len"][:])
            self.ep_offset = np.asarray(h5["ep_offset"][:])
            self.pixel_shape = tuple(h5["pixels"].shape)
            self.action_shape = tuple(h5["action"].shape)

        self.valid_eps = np.flatnonzero(self.ep_len >= self.raw_span)
        if len(self.valid_eps) < 2:
            raise ValueError(f"Need >=2 PushT episodes with >={self.raw_span} raw steps.")
        shuffled = np.random.default_rng(seed).permutation(self.valid_eps)
        split_index = min(max(int(round(len(shuffled) * train_fraction)), 1), len(shuffled) - 1)
        self.train_eps = np.sort(shuffled[:split_index])
        self.val_eps = np.sort(shuffled[split_index:])
        self.action_mean, self.action_std = self._estimate_action_block_stats(action_stats_samples)

    def _episode_pool(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train_eps
        if split == "val":
            return self.val_eps
        raise ValueError("split must be 'train' or 'val'.")

    def _estimate_action_block_stats(self, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + 11)
        blocks: list[np.ndarray] = []
        with self.h5py.File(self.path, "r") as h5:
            action = h5["action"]
            for _ in range(max(1, sample_count)):
                ep = int(rng.choice(self.train_eps))
                hi = int(self.ep_len[ep] - self.frameskip)
                rel = int(rng.integers(0, hi + 1)) if hi > 0 else 0
                start = int(self.ep_offset[ep] + rel)
                blocks.append(np.asarray(action[start : start + self.frameskip]))
        values = torch.from_numpy(np.concatenate(blocks, axis=0)).float()
        values = values[~torch.isnan(values).any(dim=1)]
        return values.mean(dim=0), values.std(dim=0).clamp_min(1e-6)

    def _window_layout(self, split: str, seq_len: int) -> tuple[np.ndarray, np.ndarray, int]:
        episodes = self._episode_pool(split)
        raw_span = self.frameskip * seq_len
        counts = np.maximum(self.ep_len[episodes] - raw_span + 1, 0).astype(np.int64)
        keep = counts > 0
        episodes = episodes[keep]
        cumulative = np.cumsum(counts[keep], dtype=np.int64)
        total = int(cumulative[-1]) if len(cumulative) else 0
        if total == 0:
            raise ValueError(f"No {split} windows available for sequence length {seq_len}.")
        return episodes, cumulative, total

    def select_window_ids(self, split: str, seq_len: int, data_fraction: float, seed: int) -> np.ndarray:
        """Deterministic subset of all valid sequence windows for a split."""

        if not 0.0 < data_fraction <= 1.0:
            raise ValueError("data_fraction must be in (0, 1].")
        _, _, total = self._window_layout(split, seq_len)
        selected = max(1, int(round(total * data_fraction)))
        if selected >= total:
            return np.arange(total, dtype=np.int64)
        return np.random.default_rng(seed).choice(total, size=selected, replace=False).astype(np.int64)

    def _resolve_windows(self, split: str, seq_len: int, window_ids: np.ndarray) -> list[tuple[int, int, int, int]]:
        episodes, cumulative, total = self._window_layout(split, seq_len)
        ids = np.asarray(window_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= total):
            raise IndexError(f"Window id outside [0, {total}) for split '{split}'.")
        positions = np.searchsorted(cumulative, ids, side="right")
        previous = np.where(positions == 0, 0, cumulative[positions - 1])
        raw_span = self.frameskip * seq_len
        windows: list[tuple[int, int, int, int]] = []
        for position, relative in zip(positions, ids - previous, strict=True):
            ep = int(episodes[position])
            start = int(self.ep_offset[ep] + relative)
            windows.append((ep, start, start + raw_span, int(self.ep_len[ep])))
        return windows

    def _load_windows(self, windows: list[tuple[int, int, int, int]], seq_len: int):
        pixels: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rows: list[dict[str, int]] = []
        with self.h5py.File(self.path, "r") as h5:
            pixels_ds = h5["pixels"]
            action_ds = h5["action"]
            for ep, start, stop, ep_len in windows:
                obs_indices = start + np.arange(seq_len) * self.frameskip
                action_blocks = [
                    np.asarray(action_ds[start + idx * self.frameskip : start + (idx + 1) * self.frameskip])
                    for idx in range(seq_len)
                ]
                pixels.append(np.asarray(pixels_ds[obs_indices]))
                actions.append(np.stack(action_blocks))
                rows.append({"episode": ep, "raw_start": start, "raw_stop": stop, "episode_len": ep_len})

        pixel_tensor = torch.from_numpy(np.stack(pixels)).permute(0, 1, 4, 2, 3).float() / 255.0
        pixel_tensor = (pixel_tensor - IMAGE_MEAN) / IMAGE_STD
        action_tensor = torch.from_numpy(np.stack(actions)).float()
        action_tensor = (action_tensor - self.action_mean.view(1, 1, 1, -1)) / self.action_std.view(1, 1, 1, -1)
        action_tensor = torch.nan_to_num(action_tensor.flatten(start_dim=2), 0.0)
        return {"pixels": pixel_tensor, "action": action_tensor}, rows

    def window_batch(self, window_ids: np.ndarray, split: str, seq_len: int):
        return self._load_windows(self._resolve_windows(split, seq_len, window_ids), seq_len)

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "frameskip": self.frameskip,
            "max_seq_len": self.max_seq_len,
            "num_episodes": int(len(self.ep_len)),
            "num_valid_episodes": int(len(self.valid_eps)),
            "num_train_episodes": int(len(self.train_eps)),
            "num_val_episodes": int(len(self.val_eps)),
            "pixel_shape": self.pixel_shape,
            "action_shape": self.action_shape,
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }


def batch_for_model(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cache_window_batches(
    sampler: PushTHDF5Sampler,
    window_ids: np.ndarray,
    split: str,
    batch_size: int,
    seq_len: int,
    count: int | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, int]]]:
    ids = np.asarray(window_ids, dtype=np.int64)
    if count is not None:
        keep = min(len(ids), count * batch_size)
        ids = np.random.default_rng(seed).choice(ids, size=keep, replace=False)
    batches: list[dict[str, torch.Tensor]] = []
    rows: list[dict[str, int]] = []
    for start in range(0, len(ids), batch_size):
        batch, batch_rows = sampler.window_batch(ids[start : start + batch_size], split, seq_len)
        batches.append(batch)
        rows.extend(batch_rows)
    return batches, rows


def training_batch_stream(
    sampler: PushTHDF5Sampler,
    window_ids: np.ndarray,
    batch_size: int,
    seq_len: int,
    seed: int,
):
    epoch = 0
    while True:
        order = np.random.default_rng(seed + epoch).permutation(window_ids)
        for start in range(0, len(order), batch_size):
            batch, _ = sampler.window_batch(order[start : start + batch_size], "train", seq_len)
            yield batch
        epoch += 1


def make_lewm_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """LeWM-style 1% linear warmup then cosine decay."""

    warmup_steps = max(1, int(0.01 * total_steps))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / float(decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


# --------------------------------------------------------------------------- #
# Objective + evaluation                                                       #
# --------------------------------------------------------------------------- #
def training_objective(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    profile: TrainingProfile,
    sigreg: torch.nn.Module,
    sigreg_weight: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    use_bf16 = profile.precision == "bf16" and device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    with autocast:
        prediction, target, emb = predict_target(model, batch, profile.history_size, profile.num_preds)
        pred_loss = F.mse_loss(prediction, target)
        sigreg_loss = sigreg(emb.transpose(0, 1))
        return pred_loss + sigreg_weight * sigreg_loss


@torch.no_grad()
def rollout_stats(model: torch.nn.Module, batch: dict[str, torch.Tensor], history_size: int, horizon: int) -> dict[str, float]:
    emb, act_emb = encode_latents(model, batch_for_model(model, batch))
    latents = emb[:, :history_size]
    predicted: list[torch.Tensor] = []
    for step in range(horizon):
        act_window = act_emb[:, step : step + history_size]
        pred_next = model.predict(latents[:, -history_size:], act_window)[:, -1:]
        predicted.append(pred_next)
        latents = torch.cat([latents, pred_next], dim=1)
    pred = torch.cat(predicted, dim=1)
    target = emb[:, history_size : history_size + horizon]
    target_norm = target.norm(dim=-1).mean()
    return {
        "rollout_mse": float(F.mse_loss(pred, target).cpu()),
        "latent_norm_drift": float((pred.norm(dim=-1).mean() - target_norm).cpu()),
        "target_latent_norm": float(target_norm.cpu()),
        "pred_latent_variance": float(pred.var(dim=(0, 1), unbiased=False).mean().cpu()),
    }


@torch.no_grad()
def average_one_step(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], history_size: int, num_preds: int) -> float:
    losses = []
    for batch in batches:
        pred, target, _ = predict_target(model, batch_for_model(model, batch), history_size, num_preds)
        losses.append(float(F.mse_loss(pred, target).cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def average_rollout(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], history_size: int, horizons: tuple[int, ...]) -> dict[str, float]:
    output: dict[str, float] = {}
    for horizon in horizons:
        stats = [rollout_stats(model, batch, history_size, horizon) for batch in batches]
        output[f"rollout_mse_{horizon}"] = float(np.mean([item["rollout_mse"] for item in stats]))
        if horizon == max(horizons):
            for key in ("latent_norm_drift", "target_latent_norm", "pred_latent_variance"):
                output[key] = float(np.mean([item[key] for item in stats]))
    return output


def evaluate(
    name: str,
    model: torch.nn.Module,
    train_batches: list[dict[str, torch.Tensor]],
    val_batches: list[dict[str, torch.Tensor]],
    profile: TrainingProfile,
    backprop_calls: int,
    train_time_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    t0 = time.perf_counter()
    train_mse = average_one_step(model, train_batches, profile.history_size, profile.num_preds)
    val_mse = average_one_step(model, val_batches, profile.history_size, profile.num_preds)
    one_step_eval_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    rollout = average_rollout(model, val_batches, profile.history_size, ROLLOUT_HORIZONS)
    rollout_eval_s = time.perf_counter() - t1
    model.train()
    row: dict[str, Any] = {
        "optimizer": name,
        "backprop_calls": int(backprop_calls),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_val_gap": val_mse - train_mse,
        "train_time_s": float(train_time_s),
        "one_step_eval_time_s": float(one_step_eval_s),
        "rollout_eval_time_s": float(rollout_eval_s),
        **rollout,
    }
    if extra:
        row.update(extra)
    return row


def mcsdca_extra(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "outer_step": info.get("outer_step"),
        "markov_chain_length": info.get("markov_chain_length"),
        "retained_samples": info.get("retained_samples"),
        "gamma_k": info.get("gamma_k"),
        "sampler_loss": info.get("loss"),
    }


# --------------------------------------------------------------------------- #
# Training loops (both driven by a shared backprop budget)                     #
# --------------------------------------------------------------------------- #
def train_baseline(
    name: str,
    model: torch.nn.Module,
    sampler: PushTHDF5Sampler,
    train_window_ids: np.ndarray,
    train_batches: list[dict[str, torch.Tensor]],
    val_batches: list[dict[str, torch.Tensor]],
    profile: TrainingProfile,
    sigreg: torch.nn.Module,
    args: argparse.Namespace,
    budget: int,
    eval_interval: int,
) -> list[dict[str, Any]]:
    params = select_full_model_parameters(model)
    optimizer = make_baseline_optimizer(name, params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_lewm_lr_scheduler(optimizer, budget)
    stream = training_batch_stream(
        sampler, train_window_ids, profile.batch_size, profile.history_size + profile.num_preds, args.seed
    )
    rows: list[dict[str, Any]] = []
    train_time_s = 0.0
    progress = tqdm(total=budget, desc=name, unit="bp", leave=False, dynamic_ncols=True)
    for step in range(1, budget + 1):
        batch = batch_for_model(model, next(stream))
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = training_objective(model, batch, profile, sigreg, args.sigreg_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, profile.gradient_clip_val)
        optimizer.step()
        scheduler.step()
        train_time_s += time.perf_counter() - t0
        progress.update(1)
        progress.set_postfix(loss=f"{float(loss.detach()):.4g}", lr=f"{scheduler.get_last_lr()[0]:.2g}")
        if step == 1 or step == budget or step % eval_interval == 0:
            row = evaluate(
                name, model, train_batches, val_batches, profile, step, train_time_s,
                {"learning_rate": scheduler.get_last_lr()[0], "status": "ok"},
            )
            rows.append(row)
            progress.set_postfix(train=f"{row['train_mse']:.4g}", val=f"{row['val_mse']:.4g}")
    progress.close()
    return rows


def train_mcsdca(
    name: str,
    model: torch.nn.Module,
    sampler: PushTHDF5Sampler,
    train_window_ids: np.ndarray,
    train_batches: list[dict[str, torch.Tensor]],
    val_batches: list[dict[str, torch.Tensor]],
    profile: TrainingProfile,
    sigreg: torch.nn.Module,
    mcsdca_config: MCSDCAConfig,
    args: argparse.Namespace,
    budget: int,
    eval_interval: int,
) -> list[dict[str, Any]]:
    optimizer_cls = MCSDCAOdLD if name == "MCSDCA-odLD" else MCSDCAUdLD
    optimizer = optimizer_cls(select_full_model_parameters(model), mcsdca_config)
    stream = training_batch_stream(
        sampler, train_window_ids, profile.batch_size, profile.history_size + profile.num_preds, args.seed
    )
    last_batch: dict[str, dict[str, torch.Tensor]] = {}

    def loss_fn() -> torch.Tensor:
        batch = batch_for_model(model, next(stream))  # fresh minibatch per inner Langevin step
        last_batch["value"] = batch
        return training_objective(model, batch, profile, sigreg, args.sigreg_weight)

    rows: list[dict[str, Any]] = []
    train_time_s = 0.0
    next_eval_at = eval_interval
    progress = tqdm(total=budget, desc=name, unit="bp", leave=False, dynamic_ncols=True)
    while optimizer.backprop_calls < budget:
        t0 = time.perf_counter()
        saved_buffers = [buf.detach().clone() for buf in model.buffers()]
        info = optimizer.step(loss_fn)
        for buf, saved in zip(model.buffers(), saved_buffers):  # discard perturbed-param BN stats
            buf.copy_(saved)
        with torch.no_grad():  # advance BN running stats once, at x_{k+1}
            training_objective(model, last_batch["value"], profile, sigreg, args.sigreg_weight)
        train_time_s += time.perf_counter() - t0
        bc = optimizer.backprop_calls
        progress.update(min(bc, budget) - progress.n)
        progress.set_postfix(outer=info["outer_step"], chain=info["markov_chain_length"], s_loss=f"{info['loss']:.4g}")
        if optimizer.outer_step == 1 or bc >= budget or bc >= next_eval_at:
            row = evaluate(
                name, model, train_batches, val_batches, profile, bc, train_time_s,
                {**mcsdca_extra(info), "status": "ok"},
            )
            rows.append(row)
            progress.set_postfix(outer=info["outer_step"], s_loss=f"{info['loss']:.4g}", val=f"{row['val_mse']:.4g}")
            next_eval_at = ((bc // eval_interval) + 1) * eval_interval
    progress.close()
    return rows


# --------------------------------------------------------------------------- #
# Result IO                                                                    #
# --------------------------------------------------------------------------- #
def save_checkpoint(run_dir: Path, name: str, model: torch.nn.Module) -> Path:
    path = run_dir / "checkpoints" / f"{optimizer_key(name)}_full_model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
    return path


def write_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_run_json(run_dir: Path, payload: dict[str, Any]) -> None:
    with (run_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    require_hdf5()
    data_path = Path(args.data_path).resolve()
    model_config_path = Path(args.model_config).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed)

    profile = resolve_profile(args)
    training_seq_len = profile.history_size + profile.num_preds
    max_seq_len = profile.history_size + max(ROLLOUT_HORIZONS)

    sampler = PushTHDF5Sampler(
        data_path, profile.frameskip, max_seq_len, args.action_stats_samples, args.seed, profile.train_fraction
    )
    model, model_config = initialize_lewm(model_config_path, device)
    base_state = clone_cpu_state(model)

    train_window_ids = sampler.select_window_ids("train", training_seq_len, profile.data_fraction, args.seed + 2)
    val_window_ids = sampler.select_window_ids("val", max_seq_len, profile.val_eval_fraction, args.seed + 7)
    steps_per_epoch = math.ceil(len(train_window_ids) / profile.batch_size)
    budget = args.backprop_budget or profile.default_budget or steps_per_epoch * profile.epochs
    eval_interval = args.eval_interval or max(1, math.ceil(budget / 10))

    train_batches, _ = cache_window_batches(
        sampler, train_window_ids, "train", profile.eval_batch_size, training_seq_len,
        count=profile.eval_train_batches, seed=args.seed + 3,
    )
    val_batches, _ = cache_window_batches(
        sampler, val_window_ids, "val", profile.eval_batch_size, max_seq_len,
        count=profile.val_batches, seed=args.seed + 4,
    )

    try:
        from stable_worldmodel.wm.loss import SIGReg
    except ImportError as exc:
        raise RuntimeError("Full LeWM training requires stable-worldmodel with SIGReg.") from exc
    sigreg = SIGReg(knots=17, num_proj=profile.sigreg_num_proj).to(device)

    mcsdca_config = resolve_mcsdca_config(profile.mcsdca, args)
    optimizers = parse_optimizers(args.optimizers)

    print(
        f"profile={args.training_profile} data_fraction={profile.data_fraction:.2%} "
        f"train_windows={len(train_window_ids):,} budget={budget:,} "
        f"(~{budget / steps_per_epoch:.1f} epochs) eval_interval={eval_interval:,}"
    )

    all_rows: list[dict[str, Any]] = []
    final_results: list[dict[str, Any]] = []

    def payload() -> dict[str, Any]:
        return {
            "args": vars(args),
            "device": str(device),
            "run_id": run_id,
            "dataset": sampler.describe(),
            "model_config": model_config,
            "profile": {"name": args.training_profile, **asdict(profile)},
            "mcsdca_config": asdict(mcsdca_config),
            "budget": int(budget),
            "steps_per_epoch": int(steps_per_epoch),
            "epochs_equivalent": budget / steps_per_epoch,
            "train_windows": int(len(train_window_ids)),
            "val_windows": int(len(val_window_ids)),
            "final_results": final_results,
        }

    for name in optimizers:
        tqdm.write(f"[{name}] training")
        seed_everything(args.seed)
        restore_state(model, base_state, device)
        try:
            if name in BASELINE_OPTIMIZERS:
                rows = train_baseline(
                    name, model, sampler, train_window_ids, train_batches, val_batches,
                    profile, sigreg, args, budget, eval_interval,
                )
            else:
                rows = train_mcsdca(
                    name, model, sampler, train_window_ids, train_batches, val_batches,
                    profile, sigreg, mcsdca_config, args, budget, eval_interval,
                )
            final = dict(rows[-1])
        except Exception as exc:  # noqa: BLE001 - a single optimizer must not kill the sweep
            traceback.print_exc()
            final = {"optimizer": name, "status": "diverged", "error": repr(exc)}
            rows = [final]
        all_rows.extend(rows)
        final_results.append(final)
        if args.save_checkpoints and final.get("status") == "ok":
            save_checkpoint(run_dir, name, model)
        write_metrics(run_dir, all_rows)
        write_run_json(run_dir, payload())
        if final.get("status") == "ok":
            tqdm.write(f"[{name}] ok val_mse={final['val_mse']:.6g} backprop_calls={final['backprop_calls']}")
        else:
            tqdm.write(f"[{name}] diverged: {final.get('error')}")

    tqdm.write(f"wrote {run_dir}")
    return final_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCSDCA vs baselines as a full-model LeWM PushT optimizer.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "pusht_expert_train.h5"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "lewm_pusht.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pusht_predictor_optimizer"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS), help="Comma list or 'all'.")
    parser.add_argument("--training-profile", choices=tuple(TRAINING_PROFILES), default="small")
    parser.add_argument("--data-fraction", type=float, default=None, help="Override profile train-window fraction.")
    parser.add_argument("--backprop-budget", type=int, default=None, help="Backward passes per optimizer (overrides profile).")
    parser.add_argument("--eval-interval", type=int, default=None, help="Backprop calls between evaluations.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None, help="Override profile precision.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override profile batch size (biggest CPU-speed knob).")
    parser.add_argument("--sigreg-num-proj", type=int, default=None, help="Override profile SIGReg projection count.")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--action-stats-samples", type=int, default=20000)
    parser.add_argument("--no-save-checkpoints", action="store_false", dest="save_checkpoints")
    parser.set_defaults(save_checkpoints=True)
    parser.add_argument("--mcsdca-epsilon", type=float, default=None)
    parser.add_argument("--mcsdca-eta", type=float, default=None, help="Overdamped Langevin step size (od_eta).")
    parser.add_argument("--mcsdca-delta", type=float, default=None, help="Underdamped Langevin step size (ud_delta).")
    parser.add_argument("--mcsdca-beta0", type=float, default=None, help="Initial DCA mixing weight 1/(1+t*gamma_0).")
    parser.add_argument("--mcsdca-langevin-steps", type=int, default=None)
    parser.add_argument("--mcsdca-burn-in", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.data_fraction is not None and not 0.0 < args.data_fraction <= 1.0:
        raise ValueError("--data-fraction must be in (0, 1].")
    if args.backprop_budget is not None and args.backprop_budget <= 0:
        raise ValueError("--backprop-budget must be positive.")
    if args.eval_interval is not None and args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive.")
    if args.sigreg_weight < 0:
        raise ValueError("--sigreg-weight must be non-negative.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.sigreg_num_proj is not None and args.sigreg_num_proj <= 0:
        raise ValueError("--sigreg-num-proj must be positive.")
    run(args)


if __name__ == "__main__":
    main()
