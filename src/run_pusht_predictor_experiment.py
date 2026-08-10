from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
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

from src.baselines import make_baseline_optimizer
from src.lewm_predictor import freeze_encoder_side, select_dynamics_head_parameters
from src.mcsdca import MCSDCAConfig, MCSDCAOdLD, MCSDCAUdLD

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
ROLLOUT_HORIZONS = (1, 3, 5)
BASELINE_OPTIMIZERS = ("AdamW", "Adam", "SGD + momentum", "RMSprop", "Adagrad")
MCSDCA_OPTIMIZERS = ("MCSDCA-odLD", "MCSDCA-udLD")
DEFAULT_OPTIMIZERS = ("AdamW", "MCSDCA-odLD", "MCSDCA-udLD")


def require_hdf5() -> Any:
    try:
        import hdf5plugin  # noqa: F401
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "PushT HDF5 pixels use Blosc compression. Install the reader with: "
            "uv pip install hdf5plugin h5py"
        ) from exc
    return h5py


def optimizer_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def parse_optimizers(value: str) -> list[str]:
    if value.lower().strip() == "all":
        return [*BASELINE_OPTIMIZERS, *MCSDCA_OPTIMIZERS]
    aliases = {optimizer_key(name): name for name in [*BASELINE_OPTIMIZERS, *MCSDCA_OPTIMIZERS]}
    selected: list[str] = []
    for raw in value.split(","):
        key = optimizer_key(raw.strip())
        if key not in aliases:
            raise ValueError(f"Unsupported optimizer '{raw}'. Supported: all, {', '.join(aliases.values())}")
        selected.append(aliases[key])
    return selected


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def torch_load(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def remap_encoder_key(key: str) -> str:
    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)$", key)
    if match:
        proj = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[match.group(2)]
        return f"encoder.layers.{match.group(1)}.attention.{proj}.{match.group(3)}"

    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)$", key)
    if match:
        return f"encoder.layers.{match.group(1)}.attention.o_proj.{match.group(2)}"

    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)$", key)
    if match:
        return f"encoder.layers.{match.group(1)}.mlp.fc1.{match.group(2)}"

    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)$", key)
    if match:
        return f"encoder.layers.{match.group(1)}.mlp.fc2.{match.group(2)}"

    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.(layernorm_before|layernorm_after)\.(weight|bias)$", key)
    if match:
        return f"encoder.layers.{match.group(1)}.{match.group(2)}.{match.group(3)}"

    return key


def remap_checkpoint_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = remap_encoder_key(key)
        if new_key in mapped:
            raise ValueError(f"Checkpoint key remap collision: {key} -> {new_key}")
        mapped[new_key] = value
    return mapped


def load_lewm_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "weights.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint weights: {weights_path}")

    config = read_json(config_path)
    model = instantiate(OmegaConf.create(config))
    state_dict = remap_checkpoint_state(torch_load(weights_path))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    freeze_encoder_side(model)
    model.train()
    model.encoder.eval()
    model.projector.eval()
    return model, config


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_state(model: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()}, strict=True)
    freeze_encoder_side(model)
    model.train()
    model.encoder.eval()
    model.projector.eval()


class PushTHDF5Sampler:
    def __init__(
        self,
        path: Path,
        frameskip: int,
        logical_seq_len: int,
        action_stats_samples: int,
        seed: int,
    ) -> None:
        self.h5py = require_hdf5()
        self.path = path
        self.frameskip = frameskip
        self.logical_seq_len = logical_seq_len
        self.raw_span = frameskip * logical_seq_len
        self.rng = np.random.default_rng(seed)

        with self.h5py.File(path, "r") as h5:
            self.ep_len = np.asarray(h5["ep_len"][:])
            self.ep_offset = np.asarray(h5["ep_offset"][:])
            self.pixel_shape = tuple(h5["pixels"].shape)
            self.action_shape = tuple(h5["action"].shape)
            self.valid_eps = np.flatnonzero(self.ep_len >= self.raw_span)
            if len(self.valid_eps) == 0:
                raise ValueError(f"No PushT episode has at least {self.raw_span} raw steps.")

        self.action_mean, self.action_std = self._estimate_action_block_stats(action_stats_samples)

    def _sample_raw_start(self) -> tuple[int, int, int, int]:
        ep = int(self.rng.choice(self.valid_eps))
        max_rel_start = int(self.ep_len[ep] - self.raw_span)
        rel_start = int(self.rng.integers(0, max_rel_start + 1)) if max_rel_start > 0 else 0
        start = int(self.ep_offset[ep] + rel_start)
        stop = start + self.raw_span
        return ep, start, stop, int(self.ep_len[ep])

    def _estimate_action_block_stats(self, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        blocks: list[np.ndarray] = []
        sample_count = max(1, sample_count)
        with self.h5py.File(self.path, "r") as h5:
            action = h5["action"]
            for _ in range(sample_count):
                _, start, _, _ = self._sample_raw_start()
                blocks.append(np.asarray(action[start : start + self.frameskip]).reshape(-1))
        values = torch.from_numpy(np.stack(blocks)).float()
        values = values[~torch.isnan(values).any(dim=1)]
        mean = values.mean(dim=0)
        std = values.std(dim=0).clamp_min(1e-6)
        return mean, std

    def sample_batch(self, batch_size: int) -> tuple[dict[str, torch.Tensor], list[dict[str, int]]]:
        pixels: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rows: list[dict[str, int]] = []
        with self.h5py.File(self.path, "r") as h5:
            pixels_ds = h5["pixels"]
            action_ds = h5["action"]
            for _ in range(batch_size):
                ep, start, stop, ep_len = self._sample_raw_start()
                obs_indices = start + np.arange(self.logical_seq_len) * self.frameskip
                action_blocks = [
                    np.asarray(action_ds[start + idx * self.frameskip : start + (idx + 1) * self.frameskip]).reshape(-1)
                    for idx in range(self.logical_seq_len)
                ]
                pixels.append(np.asarray(pixels_ds[obs_indices]))
                actions.append(np.stack(action_blocks))
                rows.append({"episode": ep, "raw_start": start, "raw_stop": stop, "episode_len": ep_len})

        pixel_tensor = torch.from_numpy(np.stack(pixels)).permute(0, 1, 4, 2, 3).float() / 255.0
        pixel_tensor = (pixel_tensor - IMAGE_MEAN) / IMAGE_STD
        action_tensor = torch.from_numpy(np.stack(actions)).float()
        action_tensor = (action_tensor - self.action_mean.view(1, 1, -1)) / self.action_std.view(1, 1, -1)
        action_tensor = torch.nan_to_num(action_tensor, 0.0)
        return {"pixels": pixel_tensor, "action": action_tensor}, rows

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "frameskip": self.frameskip,
            "logical_seq_len": self.logical_seq_len,
            "raw_span": self.raw_span,
            "num_episodes": int(len(self.ep_len)),
            "num_valid_episodes": int(len(self.valid_eps)),
            "pixel_shape": self.pixel_shape,
            "action_shape": self.action_shape,
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def encode_frozen_latents(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    model.encoder.eval()
    model.projector.eval()
    output = model.encode({"pixels": batch["pixels"].to(device)})
    return output["emb"].detach()


def cache_batches(
    model: torch.nn.Module,
    sampler: PushTHDF5Sampler,
    count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, int]]]:
    cached: list[dict[str, torch.Tensor]] = []
    sampled_rows: list[dict[str, int]] = []
    for _ in range(count):
        raw_batch, rows = sampler.sample_batch(batch_size)
        batch = move_batch(raw_batch, device)
        emb = encode_frozen_latents(model, batch, device)
        cached.append({"emb": emb, "action": batch["action"]})
        sampled_rows.extend(rows)
    return cached, sampled_rows


def one_step_loss(model: torch.nn.Module, cached_batch: dict[str, torch.Tensor], history_size: int, num_preds: int) -> torch.Tensor:
    emb = cached_batch["emb"]
    action = torch.nan_to_num(cached_batch["action"], 0.0)
    act_emb = model.action_encoder(action)
    ctx_emb = emb[:, :history_size].detach()
    ctx_act = act_emb[:, :history_size]
    target = emb[:, num_preds : num_preds + history_size].detach()
    pred = model.predict(ctx_emb, ctx_act)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(pred.shape)} != {tuple(target.shape)}")
    return F.mse_loss(pred, target)


@torch.no_grad()
def rollout_stats(
    model: torch.nn.Module,
    cached_batch: dict[str, torch.Tensor],
    history_size: int,
    horizon: int,
) -> dict[str, float]:
    emb = cached_batch["emb"]
    action = torch.nan_to_num(cached_batch["action"], 0.0)
    act_emb = model.action_encoder(action)
    latents = emb[:, :history_size].detach()
    predicted: list[torch.Tensor] = []
    for step in range(horizon):
        act_window = act_emb[:, step : step + history_size]
        pred_next = model.predict(latents[:, -history_size:], act_window)[:, -1:]
        predicted.append(pred_next)
        latents = torch.cat([latents, pred_next], dim=1)
    pred = torch.cat(predicted, dim=1)
    target = emb[:, history_size : history_size + horizon].detach()
    drift = pred.norm(dim=-1).mean() - target.norm(dim=-1).mean()
    return {
        "rollout_mse": float(F.mse_loss(pred, target).detach().cpu()),
        "latent_norm_drift": float(drift.detach().cpu()),
        "pred_latent_variance": float(pred.var(dim=(0, 1), unbiased=False).mean().detach().cpu()),
    }


@torch.no_grad()
def average_one_step(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], history_size: int, num_preds: int) -> float:
    losses = [float(one_step_loss(model, batch, history_size, num_preds).detach().cpu()) for batch in batches]
    return float(np.mean(losses))


@torch.no_grad()
def average_rollout(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    history_size: int,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for horizon in horizons:
        stats = [rollout_stats(model, batch, history_size, horizon) for batch in batches]
        output[f"rollout_mse_{horizon}"] = float(np.mean([item["rollout_mse"] for item in stats]))
        if horizon == max(horizons):
            output["latent_norm_drift"] = float(np.mean([item["latent_norm_drift"] for item in stats]))
            output["pred_latent_variance"] = float(np.mean([item["pred_latent_variance"] for item in stats]))
    return output


def evaluate(
    optimizer_name: str,
    model: torch.nn.Module,
    train_batches: list[dict[str, torch.Tensor]],
    val_batches: list[dict[str, torch.Tensor]],
    history_size: int,
    num_preds: int,
    backprop_calls: int,
    elapsed_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    train_mse = average_one_step(model, train_batches, history_size, num_preds)
    val_mse = average_one_step(model, val_batches, history_size, num_preds)
    rollout = average_rollout(model, val_batches, history_size, ROLLOUT_HORIZONS)
    model.train()
    model.encoder.eval()
    model.projector.eval()
    result: dict[str, Any] = {
        "optimizer": optimizer_name,
        "backprop_calls": int(backprop_calls),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_val_gap": val_mse - train_mse,
        "time_s": float(elapsed_s),
        **rollout,
    }
    if extra:
        result.update(extra)
    return result


def next_batch(batches: list[dict[str, torch.Tensor]], index: int) -> dict[str, torch.Tensor]:
    return batches[index % len(batches)]


def train_baseline(
    name: str,
    model: torch.nn.Module,
    train_batches: list[dict[str, torch.Tensor]],
    val_batches: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimizer = make_baseline_optimizer(
        name,
        select_dynamics_head_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    metric_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, args.backprop_budget + 1):
        batch = next_batch(train_batches, step - 1)
        optimizer.zero_grad(set_to_none=True)
        loss = one_step_loss(model, batch, args.history_size, args.num_preds)
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.backprop_budget or step % args.eval_interval == 0:
            metric_rows.append(evaluate(name, model, train_batches, val_batches, args.history_size, args.num_preds, step, time.perf_counter() - start))
    return metric_rows[-1], metric_rows


def train_mcsdca(
    name: str,
    model: torch.nn.Module,
    train_batches: list[dict[str, torch.Tensor]],
    val_batches: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimizer_cls = MCSDCAOdLD if name == "MCSDCA-odLD" else MCSDCAUdLD
    optimizer = optimizer_cls(select_dynamics_head_parameters(model), MCSDCAConfig())
    metric_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    outer_index = 0
    last_info: dict[str, Any] = {}
    while optimizer.backprop_calls < args.backprop_budget:
        batch = next_batch(train_batches, outer_index)
        last_info = optimizer.step(lambda: one_step_loss(model, batch, args.history_size, args.num_preds))
        outer_index += 1
        if (
            optimizer.backprop_calls >= args.backprop_budget
            or optimizer.backprop_calls == last_info["backprop_calls"]
            or optimizer.backprop_calls % args.eval_interval <= int(last_info["markov_chain_length"])
        ):
            metric_rows.append(
                evaluate(
                    name,
                    model,
                    train_batches,
                    val_batches,
                    args.history_size,
                    args.num_preds,
                    optimizer.backprop_calls,
                    time.perf_counter() - start,
                    {
                        "outer_step": last_info.get("outer_step"),
                        "markov_chain_length": last_info.get("markov_chain_length"),
                        "retained_samples": last_info.get("retained_samples"),
                        "gamma_k": last_info.get("gamma_k"),
                        "sampler_loss": last_info.get("loss"),
                    },
                )
            )
    return metric_rows[-1], metric_rows


def predictor_side_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    prefixes = ("action_encoder.", "predictor.", "pred_proj.")
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if key.startswith(prefixes)}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def make_tables(final_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictor_rows = [
        {
            "Optimizer": item["optimizer"],
            "Backprop calls": item["backprop_calls"],
            "Train MSE": item["train_mse"],
            "Val MSE": item["val_mse"],
            "Train/Val gap": item["train_val_gap"],
            "Latent drift": item["latent_norm_drift"],
            "Time": item["time_s"],
        }
        for item in final_results
    ]
    result_by_name = {item["optimizer"]: item for item in final_results}
    rollout_rows = []
    for name in ["AdamW", "MCSDCA-odLD", "MCSDCA-udLD"]:
        if name not in result_by_name:
            continue
        item = result_by_name[name]
        rollout_rows.append(
            {
                "Optimizer": name,
                "Horizon": max(ROLLOUT_HORIZONS),
                "Rollout MSE@1": item.get("rollout_mse_1"),
                "Rollout MSE@3": item.get("rollout_mse_3"),
                "Rollout MSE@5": item.get("rollout_mse_5"),
                "Latent drift": item.get("latent_norm_drift"),
                "Time": item.get("time_s"),
            }
        )
    planning_rows = [
        {"Optimizer": name, "CEM horizon": "TBD", "Eval episodes": "TBD", "PushT score": "TBD", "CEM cost": "TBD", "Eval time": "Skipped"}
        for name in ["AdamW", "MCSDCA-odLD", "MCSDCA-udLD"]
        if name in result_by_name
    ]
    return predictor_rows, rollout_rows, planning_rows


def run(args: argparse.Namespace) -> Path:
    h5py = require_hdf5()
    del h5py
    data_path = Path(args.data_path).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    max_horizon = max(ROLLOUT_HORIZONS)
    logical_seq_len = args.history_size + max_horizon
    sampler = PushTHDF5Sampler(data_path, args.frameskip, logical_seq_len, args.action_stats_samples, args.seed)

    print(f"Loading checkpoint from {checkpoint_dir}")
    model, checkpoint_config = load_lewm_checkpoint(checkpoint_dir, device)
    base_state = clone_cpu_state(model)

    print("Caching frozen train/val latents")
    train_batches, train_rows = cache_batches(model, sampler, args.train_batches, args.batch_size, device)
    val_batches, val_rows = cache_batches(model, sampler, args.val_batches, args.batch_size, device)

    optimizers = parse_optimizers(args.optimizers)
    final_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for name in optimizers:
        print(f"Training {name}")
        restore_state(model, base_state, device)
        if name in BASELINE_OPTIMIZERS:
            final, rows = train_baseline(name, model, train_batches, val_batches, args)
        else:
            final, rows = train_mcsdca(name, model, train_batches, val_batches, args)
        final_results.append(final)
        metric_rows.extend(rows)
        if args.save_checkpoints:
            safe_name = optimizer_key(name)
            checkpoint_out = run_dir / "checkpoints" / f"{safe_name}_predictor_side.pt"
            checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(predictor_side_state(model), checkpoint_out)
        print(f"Finished {name}: val_mse={final['val_mse']:.6g}, backprop_calls={final['backprop_calls']}")

    predictor_rows, rollout_rows, planning_rows = make_tables(final_results)

    config_out = {
        "args": vars(args),
        "device": str(device),
        "run_id": run_id,
        "dataset": sampler.describe(),
        "checkpoint_config": checkpoint_config,
        "mcsdca_default_config": asdict(MCSDCAConfig()),
        "train_sample_rows": train_rows[:20],
        "val_sample_rows": val_rows[:20],
    }
    summary = {"run_dir": str(run_dir), "final_results": final_results, "tables": {"predictor": predictor_rows, "rollout": rollout_rows, "planning": planning_rows}}

    write_json(run_dir / "config.json", config_out)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "metrics_step.csv", metric_rows, sorted({key for row in metric_rows for key in row.keys()}))
    write_csv(run_dir / "predictor_table.csv", predictor_rows, list(predictor_rows[0].keys()) if predictor_rows else [])
    write_csv(run_dir / "rollout_table.csv", rollout_rows, list(rollout_rows[0].keys()) if rollout_rows else [])
    write_csv(run_dir / "planning_table.csv", planning_rows, list(planning_rows[0].keys()) if planning_rows else [])

    print(f"Wrote results to {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate MCSDCA as LeWM PushT predictor optimizer.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "pusht_expert_train.h5"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "data" / "checkpoints" / "pusht" / "lewm"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pusht_predictor_optimizer"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS), help="Comma list or 'all'.")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--backprop-budget", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--action-stats-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--no-save-checkpoints", action="store_false", dest="save_checkpoints")
    parser.set_defaults(save_checkpoints=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.train_batches <= 0 or args.val_batches <= 0 or args.batch_size <= 0:
        raise ValueError("train-batches, val-batches, and batch-size must be positive.")
    if args.backprop_budget <= 0:
        raise ValueError("backprop-budget must be positive.")
    if args.eval_interval <= 0:
        raise ValueError("eval-interval must be positive.")
    run(args)


if __name__ == "__main__":
    main()
