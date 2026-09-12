"""Engine: train and evaluate MCSDCA-odLD / MCSDCA-udLD against an AdamW baseline
as a full-model LeWM PushT predictor optimizer.

All five LeWM modules (encoder, projector, action_encoder, predictor, pred_proj)
are initialized from scratch and trained jointly on
``prediction MSE + sigreg_weight * SIGReg`` (the LeWM objective). Every optimizer
gets the same initial weights and the same batch stream. The training budget is
``epochs * steps_per_epoch`` backward passes (the MCSDCA paper's ``40 * N`` when
``epochs = 40``); MCSDCA spends ``~n_k`` of those per outer step.

Every knob comes from a single YAML (``configs/experiment1.yaml``) -- there is no
tuning. ``src/run_experiment1.py`` is the driver that sweeps data fractions x
seeds; this module's ``main()`` runs one (fraction, seed). Results land in
``outputs/<dataTAG>/<stamp>__seed<seed>/`` (one fresh folder per run, never
overwritten). Planning/CEM evaluation lives in ``evaluate_planning.py``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
import warnings
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

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
BASELINE_OPTIMIZERS = ("AdamW", "Adam")
MCSDCA_OPTIMIZERS = ("MCSDCA-odLD", "MCSDCA-udLD")
DEFAULT_OPTIMIZERS = ("AdamW", "MCSDCA-odLD", "MCSDCA-udLD")
# Measured on real PushT frames (150,528 raw bytes/frame vs ~11,532 PNG bytes/frame at
# compress_level=6): used only for capacity planning in resolve_cache_placement -- the
# actual ratio realized during a build is printed precisely by PushTHDF5Sampler._build_png_blob.
PNG_COMPRESSION_RATIO_ESTIMATE = 13.0


# --------------------------------------------------------------------------- #
# Training profile (one instance, built from the YAML config)                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingProfile:
    data_fraction: float
    epochs: int  # budget = epochs * steps_per_epoch (MCSDCA paper's 40*N at epochs=40)
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
    gradient_clip_val: float = 1.0  # AdamW baseline only (le-wm/config/train/lewm.yaml); MCSDCA never clips
    train_fraction: float = 0.9  # episode-level train/val split
    val_eval_fraction: float = 0.15  # fraction of val windows kept for evaluation
    default_budget: int | None = None  # overrides steps_per_epoch * epochs when set


def mcsdca_config_from_cfg(cfg: Any) -> MCSDCAConfig:
    """Build the MCSDCA config straight from the YAML ``mcsdca:`` block."""

    m = cfg.mcsdca
    out = MCSDCAConfig(
        langevin_steps=int(m.langevin_steps),
        langevin_steps_rate=float(m.langevin_steps_rate),
        max_langevin_steps=(None if m.max_langevin_steps in (None, "null") else int(m.max_langevin_steps)),
        burn_in=int(m.burn_in),
        local_entropy_time=float(m.local_entropy_time),
        gamma=float(m.gamma),
        gamma_power=float(m.gamma_power),
        beta0=(None if m.beta0 in (None, "null") else float(m.beta0)),
        epsilon=float(m.epsilon),
        eta=float(m.eta),
        delta=float(m.delta),
    )
    out.validate()
    return out


def profile_from_args(args: Any) -> TrainingProfile:
    """Build the single TrainingProfile from a flattened run-args namespace."""

    budget = getattr(args, "budget", None)
    return TrainingProfile(
        data_fraction=float(args.data_fraction),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        eval_train_batches=int(args.eval_train_batches),
        val_batches=int(args.val_batches),
        precision=str(args.precision),
        sigreg_num_proj=int(args.sigreg_num_proj),
        mcsdca=args.mcsdca_config,
        history_size=int(args.history_size),
        num_preds=int(args.num_preds),
        frameskip=int(args.frameskip),
        gradient_clip_val=float(args.gradient_clip_val),
        train_fraction=float(args.train_fraction),
        val_eval_fraction=float(args.val_eval_fraction),
        default_budget=(None if budget in (None, "null") else int(budget)),
    )


def build_run_args(cfg: Any, *, data_fraction: float, seed: int) -> SimpleNamespace:
    """Flatten the merged YAML config into the namespace ``run()`` consumes for
    one (data_fraction, seed). Keeps the data-loading / cache fields verbatim."""

    t, e, c = cfg.train, cfg.eval, cfg.cache
    device = str(cfg.device)
    return SimpleNamespace(
        # data / model
        data_path=str(cfg.data.path),
        model=OmegaConf.to_container(cfg.model, resolve=True),
        data_fraction=float(data_fraction),
        seed=int(seed),
        run_tag=str(getattr(cfg, "run_tag", "") or ""),
        device=device,
        num_threads=int(cfg.num_threads),
        action_stats_samples=int(cfg.data.action_stats_samples),
        train_fraction=float(cfg.data.train_fraction),
        val_eval_fraction=float(cfg.data.val_eval_fraction),
        # training profile knobs
        epochs=int(t.epochs),
        budget=(None if t.budget in (None, "null") else int(t.budget)),
        batch_size=int(t.batch_size),
        precision=str(t.precision),
        history_size=int(t.history_size),
        num_preds=int(t.num_preds),
        frameskip=int(t.frameskip),
        gradient_clip_val=float(t.gradient_clip_val),
        eval_batch_size=int(e.batch_size),
        eval_train_batches=int(e.train_batches),
        val_batches=int(e.val_batches),
        eval_interval=None,
        eval_interval_frac=float(t.eval_interval_frac),
        eval_every_epochs=(None if t.eval_every_epochs in (None, "null") else int(t.eval_every_epochs)),
        early_stop_enabled=bool(t.early_stop.enabled),
        early_stop_patience=int(t.early_stop.patience_evals),
        early_stop_min_delta=float(t.early_stop.min_delta),
        # optimizers
        optimizers=list(cfg.optimizers),
        lr=float(t.adamw.lr),
        weight_decay=float(t.adamw.weight_decay),
        sigreg_weight=float(cfg.loss.sigreg.weight),
        sigreg_num_proj=int(cfg.loss.sigreg.num_proj),
        sigreg_knots=int(cfg.loss.sigreg.knots),
        mcsdca_config=mcsdca_config_from_cfg(cfg),
        config_yaml=OmegaConf.to_yaml(cfg),  # verbatim snapshot -> written into each run folder
        # window cache (unchanged semantics)
        window_cache=str(c.mode),
        cache_max_gb=float(c.max_gb),
        cache_gpu_reserve_gb=float(c.gpu_reserve_gb),
        cache_ram_gb=float(c.ram_gb),
        cache_disk_gb=float(c.disk_gb),
        memmap_dir=(None if c.memmap_dir in (None, "null") else str(c.memmap_dir)),
        prefetch_depth=int(c.prefetch_depth),
        prefetch_workers=int(c.prefetch_workers),
        cache_memo_size=int(c.memo_size),
        png_compress_level=int(c.png_compress_level),
        png_build_workers=int(c.png_build_workers),
        # io
        save_checkpoints=bool(cfg.save_checkpoints),
    )


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


def require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "The PNG-blob window cache (cache.mode=png) needs Pillow. Install: uv pip install pillow"
        ) from exc
    return Image


def optimizer_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def parse_optimizers(value: Any) -> list[str]:
    names = [*BASELINE_OPTIMIZERS, *MCSDCA_OPTIMIZERS]
    aliases = {optimizer_key(name): name for name in names}
    raw_list = value if isinstance(value, (list, tuple)) else str(value).split(",")
    selected: list[str] = []
    for raw in raw_list:
        key = optimizer_key(str(raw).strip())
        if key not in aliases:
            raise ValueError(f"Unsupported optimizer '{raw}'. Options: {', '.join(names)}")
        selected.append(aliases[key])
    return selected


def initialize_lewm(model_cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate the LeWM model from an inline (``_target_``-style) config dict."""

    model = instantiate(OmegaConf.create(model_cfg)).to(device)
    enable_full_model_training(model)
    return model, model_cfg


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


_COMPUTE_TUNED = False


def tune_compute(device: torch.device, num_threads: int = 0) -> None:
    """Idempotent process-wide compute knobs (plan 5, no torch.compile).

    On CUDA: enable TF32 matmul + cuDNN autotune (2-4x on Ampere+, negligible
    accuracy impact for this training). Always: cap intra-op threads so the
    CPU-side gather / normalize / numpy work uses cores without oversubscribing.
    """

    global _COMPUTE_TUNED
    threads = num_threads if num_threads and num_threads > 0 else max(1, min(16, os.cpu_count() or 8))
    torch.set_num_threads(threads)
    if _COMPUTE_TUNED:
        return
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # noqa: BLE001 - older torch
            pass
    _COMPUTE_TUNED = True


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

        self._h5 = None
        self._pixels_ds = None
        self._load_or_fetch_metadata()

        self.valid_eps = np.flatnonzero(self.ep_len >= self.raw_span)
        if len(self.valid_eps) < 2:
            raise ValueError(f"Need >=2 PushT episodes with >={self.raw_span} raw steps.")
        shuffled = np.random.default_rng(seed).permutation(self.valid_eps)
        split_index = min(max(int(round(len(shuffled) * train_fraction)), 1), len(shuffled) - 1)
        self.train_eps = np.sort(shuffled[:split_index])
        self.val_eps = np.sort(shuffled[split_index:])
        self.action_mean, self.action_std = self._estimate_action_block_stats(action_stats_samples)

    def _metadata_cache_path(self) -> Path:
        # Fixed default location, independent of --set cache.memmap_dir -- this
        # sidecar is tiny (~19 MB, dominated by the action array) and tied to
        # the dataset itself, not to any particular resident-cache tier.
        return Path(self.path).parent / ".framecache" / "dataset_meta.npz"

    def _load_or_fetch_metadata(self) -> None:
        """Populate ep_len/ep_offset/pixel_shape/action_shape/_actions_all/_mtime.

        Prefers a small persisted sidecar (``dataset_meta.npz``) so that once a
        PNG/memmap pixel cache is fully built, the (much bigger) original
        ``--data`` file is no longer needed at all -- safe to delete it to save
        disk. Falls back to the real HDF5 file when no sidecar exists yet, and
        opportunistically writes one so the next run can skip it."""

        meta_path = self._metadata_cache_path()
        if meta_path.exists():
            data = np.load(meta_path, allow_pickle=False)
            self._mtime = int(data["mtime"])
            self.ep_len = data["ep_len"]
            self.ep_offset = data["ep_offset"]
            self.pixel_shape = tuple(int(v) for v in data["pixel_shape"])
            self.action_shape = tuple(int(v) for v in data["action_shape"])
            self._actions_all = data["actions_all"].astype(np.float32)
            return

        if not Path(self.path).exists():
            raise FileNotFoundError(
                f"Source file missing ({self.path}) and no metadata cache at {meta_path}. "
                "Episode lengths/offsets and actions live only in the original --data file "
                "the first time; restore it once so this sidecar can be written, then it is "
                "safe to delete again."
            )

        # One long-lived read handle (a bigger chunk cache than the default 1 MB
        # so repeated slab reads during cache warm-up stay in memory). The
        # ``action`` dataset is tiny (~19 MB) so we pull it fully into RAM once;
        # every per-window action slice is then a pure numpy view.
        self._h5 = self.h5py.File(self.path, "r", rdcc_nbytes=256 * 1024 * 1024)
        self._pixels_ds = self._h5["pixels"]
        self._mtime = int(Path(self.path).stat().st_mtime)
        self.ep_len = np.asarray(self._h5["ep_len"][:])
        self.ep_offset = np.asarray(self._h5["ep_offset"][:])
        self.pixel_shape = tuple(self._h5["pixels"].shape)
        self.action_shape = tuple(self._h5["action"].shape)
        self._actions_all = np.asarray(self._h5["action"][:], dtype=np.float32)

        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                meta_path, mtime=self._mtime, ep_len=self.ep_len, ep_offset=self.ep_offset,
                pixel_shape=np.asarray(self.pixel_shape), action_shape=np.asarray(self.action_shape),
                actions_all=self._actions_all,
            )
        except OSError:
            pass  # best effort -- a full disk here shouldn't block training

    @property
    def pixels_ds(self) -> Any:
        """The raw HDF5 pixel dataset -- only needed for streaming, or to build
        a cache tier that isn't already on disk. Raises a clear error instead of
        a confusing ``NoneType`` crash when the source file was deleted after
        every pixel cache was already built."""

        if self._pixels_ds is None:
            raise RuntimeError(
                f"Need the original HDF5 pixels ({self.path}) for this operation (no existing "
                "PNG/memmap cache covers it), but the source file is missing and only the "
                "metadata sidecar was found. Restore the --data file to build this selection."
            )
        return self._pixels_ds

    def _episode_pool(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train_eps
        if split == "val":
            return self.val_eps
        raise ValueError("split must be 'train' or 'val'.")

    def close(self) -> None:
        handle = getattr(self, "_h5", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - best effort on teardown
                pass
            self._h5 = None

    def __del__(self) -> None:
        self.close()

    def _estimate_action_block_stats(self, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + 11)
        # ``_actions_all`` is already resident; sample block start offsets and
        # gather with fancy indexing instead of thousands of h5py reads.
        starts: list[int] = []
        for _ in range(max(1, sample_count)):
            ep = int(rng.choice(self.train_eps))
            hi = int(self.ep_len[ep] - self.frameskip)
            rel = int(rng.integers(0, hi + 1)) if hi > 0 else 0
            starts.append(int(self.ep_offset[ep] + rel))
        idx = np.asarray(starts, dtype=np.int64)[:, None] + np.arange(self.frameskip, dtype=np.int64)[None, :]
        values = torch.from_numpy(self._actions_all[idx.reshape(-1)]).float()
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

    def _normalize_actions(self, raw: np.ndarray, seq_len: int) -> torch.Tensor:
        """``raw`` is ``[N, seq_len * frameskip, A]`` -> ``[N, seq_len, frameskip*A]``
        normalized exactly as the original ``_load_windows`` did."""

        n, a = raw.shape[0], raw.shape[-1]
        act = torch.from_numpy(np.ascontiguousarray(raw)).float().view(n, seq_len, self.frameskip, a)
        act = (act - self.action_mean.view(1, 1, 1, -1)) / self.action_std.view(1, 1, 1, -1)
        return torch.nan_to_num(act.flatten(start_dim=2), 0.0)

    def _load_windows(self, windows: list[tuple[int, int, int, int]], seq_len: int):
        pixels_ds = self.pixels_ds
        pixels: list[np.ndarray] = []
        action_starts: list[int] = []
        rows: list[dict[str, int]] = []
        raw_span = self.frameskip * seq_len
        for ep, start, stop, ep_len in windows:
            obs_indices = start + np.arange(seq_len) * self.frameskip
            pixels.append(np.asarray(pixels_ds[obs_indices]))
            action_starts.append(start)
            rows.append({"episode": ep, "raw_start": start, "raw_stop": stop, "episode_len": ep_len})

        pixel_tensor = torch.from_numpy(np.stack(pixels)).permute(0, 1, 4, 2, 3).float() / 255.0
        pixel_tensor = (pixel_tensor - IMAGE_MEAN) / IMAGE_STD
        gather = np.asarray(action_starts, dtype=np.int64)[:, None] + np.arange(raw_span, dtype=np.int64)[None, :]
        action_tensor = self._normalize_actions(self._actions_all[gather], seq_len)
        return {"pixels": pixel_tensor, "action": action_tensor}, rows

    def window_batch(self, window_ids: np.ndarray, split: str, seq_len: int):
        return self._load_windows(self._resolve_windows(split, seq_len, window_ids), seq_len)

    # ------------------------------------------------------------------ #
    # Resident-cache construction (see WindowCache)                      #
    # ------------------------------------------------------------------ #
    def _window_frame_ids(self, windows: list[tuple[int, int, int, int]], seq_len: int) -> np.ndarray:
        """``[N, seq_len]`` global pixel-frame indices for a resolved window list."""

        starts = np.fromiter((w[1] for w in windows), dtype=np.int64, count=len(windows))
        return starts[:, None] + np.arange(seq_len, dtype=np.int64)[None, :] * self.frameskip

    def count_unique_frames(self, split: str, seq_len: int, window_ids: np.ndarray) -> int:
        windows = self._resolve_windows(split, seq_len, window_ids)
        return int(np.unique(self._window_frame_ids(windows, seq_len)).size)

    @staticmethod
    def _contiguous_runs(uniq: np.ndarray):
        """Yield ``(lo_pos, hi_pos, first_frame_id)`` for each maximal run of
        consecutive frame ids in the SORTED unique array ``uniq`` -- shared by
        every cache builder that wants big slab reads instead of one read per
        frame (raw memmap and PNG-blob alike)."""

        boundaries = np.flatnonzero(np.diff(uniq) != 1)
        run_edges = np.concatenate(([0], boundaries + 1, [uniq.size]))
        for lo_pos, hi_pos in zip(run_edges[:-1], run_edges[1:], strict=True):
            yield int(lo_pos), int(hi_pos), int(uniq[lo_pos])  # run is contiguous: uniq[lo:hi] == first+arange(...)

    def _fill_frames(self, dst: Any, uniq: np.ndarray, read_chunk: int) -> None:
        """Read the unique frames ``uniq`` (sorted) from HDF5 in contiguous runs
        into ``dst`` (a ``[F, C, H, W]`` torch tensor slice-assignable, or an
        ``np.memmap``). One big slab read per run, permuted H,W,C -> C,H,W."""

        is_tensor = isinstance(dst, torch.Tensor)
        for lo_pos, hi_pos, first in self._contiguous_runs(uniq):
            for sub in range(lo_pos, hi_pos, read_chunk):
                sub_hi = min(sub + read_chunk, hi_pos)
                slab = self.pixels_ds[first + (sub - lo_pos) : first + (sub_hi - lo_pos)]  # [n, H, W, C] uint8
                chw = np.ascontiguousarray(np.moveaxis(slab, 3, 1))  # [n, C, H, W]
                dst[sub:sub_hi] = torch.from_numpy(chw) if is_tensor else chw

    def _window_actions(self, windows: list[tuple[int, int, int, int]], seq_len: int) -> torch.Tensor:
        raw_span = self.frameskip * seq_len
        gather = (
            np.fromiter((w[1] for w in windows), dtype=np.int64, count=len(windows))[:, None]
            + np.arange(raw_span, dtype=np.int64)[None, :]
        )
        return self._normalize_actions(self._actions_all[gather], seq_len)

    def materialize_windows(
        self,
        split: str,
        seq_len: int,
        window_ids: np.ndarray,
        frames_device: torch.device,
        compute_device: torch.device,
        read_chunk: int = 1024,
    ) -> "WindowCache":
        """Decode every requested window once into a resident ``[F, C, H, W]``
        ``uint8`` tensor on ``frames_device`` (GPU or CPU RAM). Per-batch
        gathering + float normalization then happens on ``compute_device`` with
        no further disk access."""

        windows = self._resolve_windows(split, seq_len, window_ids)
        frame_ids = self._window_frame_ids(windows, seq_len)
        uniq = np.unique(frame_ids)
        _, height, width, channels = self.pixel_shape

        frames_t = torch.empty((uniq.size, channels, height, width), dtype=torch.uint8, device=frames_device)
        self._fill_frames(frames_t, uniq, read_chunk)
        local_rows = np.searchsorted(uniq, frame_ids).astype(np.int64)

        return WindowCache(
            frames_u8=frames_t,
            window_frame_rows=torch.from_numpy(local_rows).to(frames_device),
            window_actions=self._window_actions(windows, seq_len),  # small; moved to device per-batch
            seq_len=seq_len,
            compute_device=compute_device,
            backing=frames_device.type,
        )

    def _pixel_cache_tag(self, split: str, seq_len: int, window_ids: np.ndarray) -> tuple[np.ndarray, str]:
        """``(uniq, tag)`` shared by every disk cache tier (memmap, png) and by
        ``existing_cache_gb`` -- same content hash, so tag collisions/misses
        stay consistent across tiers and across a size-only existence check."""

        windows = self._resolve_windows(split, seq_len, window_ids)
        frame_ids = self._window_frame_ids(windows, seq_len)
        uniq = np.unique(frame_ids)
        _, height, width, channels = self.pixel_shape
        tag = hashlib.sha1(
            repr((str(self.path), self._mtime, split, seq_len, (int(uniq.size), channels, height, width),
                  int(uniq[0]), int(uniq[-1]), int(len(window_ids)))).encode()
        ).hexdigest()[:16]
        return uniq, tag

    def existing_cache_gb(self, split: str, seq_len: int, window_ids: np.ndarray, cache_dir: Path) -> dict[str, float]:
        """Which disk cache tiers already have a file matching this EXACT
        selection sitting in ``cache_dir``, keyed by tier name -> size in GB.

        Lets ``resolve_cache_placement`` tell "already built, just reuse it"
        apart from "need to write N new GB" -- a free-disk-space check against
        the SIZE OF A REBUILD is wrong once a matching cache already exists,
        since reusing it needs zero new bytes (this is what previously made a
        fully-built PNG cache get rejected once it ate into the free-space
        margin that same build had used)."""

        uniq, tag = self._pixel_cache_tag(split, seq_len, window_ids)
        _, height, width, channels = self.pixel_shape
        cache_dir = Path(cache_dir)
        found: dict[str, float] = {}

        dat = cache_dir / f"frames_{split}_{tag}.dat"
        meta = dat.with_suffix(".json")
        want_mm = {"shape": [int(uniq.size), channels, height, width], "dtype": "uint8"}
        if dat.exists() and meta.exists():
            try:
                if json.loads(meta.read_text(encoding="utf-8")) == want_mm:
                    found["memmap"] = dat.stat().st_size / 1e9
            except (OSError, ValueError):
                pass

        blob = cache_dir / f"frames_{split}_{tag}.png.blob"
        idx = blob.with_suffix(".idx.npy")
        pmeta = blob.with_suffix(".json")
        if blob.exists() and idx.exists() and pmeta.exists():
            try:
                if json.loads(pmeta.read_text(encoding="utf-8")).get("count") == int(uniq.size):
                    found["png"] = blob.stat().st_size / 1e9
            except (OSError, ValueError):
                pass

        return found

    def materialize_windows_memmap(
        self,
        split: str,
        seq_len: int,
        window_ids: np.ndarray,
        memmap_dir: Path,
        compute_device: torch.device,
        read_chunk: int = 2048,
    ) -> "WindowCache":
        """Like ``materialize_windows`` but the unique frames live in an on-disk
        ``uint8`` memmap (NVMe). The OS page cache keeps the hot working set in
        RAM after the first epoch; the decode pass runs once and is reused across
        sweep points / reruns via a content hash."""

        uniq, tag = self._pixel_cache_tag(split, seq_len, window_ids)
        windows = self._resolve_windows(split, seq_len, window_ids)
        frame_ids = self._window_frame_ids(windows, seq_len)
        _, height, width, channels = self.pixel_shape
        shape = (int(uniq.size), channels, height, width)

        Path(memmap_dir).mkdir(parents=True, exist_ok=True)
        dat = Path(memmap_dir) / f"frames_{split}_{tag}.dat"
        meta = dat.with_suffix(".json")
        want = {"shape": list(shape), "dtype": "uint8"}
        ready = dat.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == want

        if ready:
            frames_np = np.memmap(dat, dtype=np.uint8, mode="r", shape=shape)
        else:
            frames_np = np.memmap(dat, dtype=np.uint8, mode="w+", shape=shape)
            self._fill_frames(frames_np, uniq, read_chunk)
            frames_np.flush()
            meta.write_text(json.dumps(want), encoding="utf-8")

        local_rows = np.searchsorted(uniq, frame_ids).astype(np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only memmap -> "non-writable tensor" notice
            frames_t = torch.from_numpy(frames_np)

        return WindowCache(
            frames_u8=frames_t,
            window_frame_rows=torch.from_numpy(local_rows),
            window_actions=self._window_actions(windows, seq_len),  # kept on CPU; moved per-batch
            seq_len=seq_len,
            compute_device=compute_device,
            backing="memmap",
            disk_path=dat,
        )

    def _build_png_blob(
        self, uniq: np.ndarray, blob_path: Path, compress_level: int, workers: int, read_chunk: int,
    ) -> np.ndarray:
        """Encode every frame in ``uniq`` to a lossless PNG and append it to one
        blob file; returns the ``[F, 2]`` int64 ``(offset, length)`` index.

        Reads big contiguous slabs from HDF5 (via ``_contiguous_runs``, same as
        the raw memmap builder) and PNG-encodes each frame in a thread pool --
        Pillow's encoder releases the GIL, so this scales with real cores
        (measured ~6x at 16 threads on an 8-core box)."""

        Image = require_pillow()
        n = int(uniq.size)
        index = np.empty((n, 2), dtype=np.int64)

        def encode_one(hwc: np.ndarray) -> bytes:
            buf = io.BytesIO()
            Image.fromarray(hwc).save(buf, format="PNG", compress_level=compress_level)
            return buf.getvalue()

        print(f"[png-cache] building {blob_path.name}  ({n:,} unique frames, {workers} workers, "
              f"compress_level={compress_level})...")
        t_start = time.perf_counter()
        offset = 0
        pos = 0
        with open(blob_path, "wb") as out, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            progress = tqdm(total=n, desc="png-cache", unit="frame", dynamic_ncols=True)
            for lo_pos, hi_pos, first in self._contiguous_runs(uniq):
                for sub in range(lo_pos, hi_pos, read_chunk):
                    sub_hi = min(sub + read_chunk, hi_pos)
                    slab = np.asarray(self.pixels_ds[first + (sub - lo_pos) : first + (sub_hi - lo_pos)])
                    for blob in pool.map(encode_one, slab):  # pool.map preserves input order
                        index[pos] = (offset, len(blob))
                        out.write(blob)
                        offset += len(blob)
                        pos += 1
                    elapsed = time.perf_counter() - t_start
                    progress.set_postfix(fps=f"{pos / elapsed:,.0f}" if elapsed > 0 else "0")
                    progress.update(sub_hi - sub)
            progress.close()

        elapsed = time.perf_counter() - t_start
        size_gb = offset / 1e9
        _, height, width, channels = self.pixel_shape
        raw_gb = n * height * width * channels / 1e9
        ratio = raw_gb / size_gb if size_gb > 0 else 0.0
        print(f"[png-cache] done: {n:,} frames, {size_gb:.2f} GB (raw would be ~{raw_gb:.1f} GB, "
              f"{ratio:.1f}x smaller), {elapsed:.1f}s ({n / elapsed:,.0f} fps)")
        return index

    def materialize_windows_png(
        self,
        split: str,
        seq_len: int,
        window_ids: np.ndarray,
        cache_dir: Path,
        compute_device: torch.device,
        compress_level: int = 6,
        build_workers: int = 0,
        read_chunk: int = 2048,
    ) -> "WindowCache":
        """Like ``materialize_windows_memmap`` but each unique frame is stored as
        an independently-seekable, lossless PNG blob (~13x smaller on disk than
        raw uint8) instead of a fixed-stride memmap. Built once and reused across
        reruns via the same content-hash scheme; see ``_build_png_blob``."""

        require_pillow()
        uniq, tag = self._pixel_cache_tag(split, seq_len, window_ids)
        windows = self._resolve_windows(split, seq_len, window_ids)
        frame_ids = self._window_frame_ids(windows, seq_len)
        _, height, width, channels = self.pixel_shape
        n_unique = int(uniq.size)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        blob_path = Path(cache_dir) / f"frames_{split}_{tag}.png.blob"
        idx_path = blob_path.with_suffix(".idx.npy")
        meta_path = blob_path.with_suffix(".json")
        want = {"count": n_unique, "compress_level": int(compress_level)}
        ready = (
            blob_path.exists() and idx_path.exists() and meta_path.exists()
            and json.loads(meta_path.read_text(encoding="utf-8")) == want
        )

        if ready:
            print(f"[png-cache] found cached blob, reuse (skip build): {blob_path.name}")
            png_index = np.load(idx_path)
        else:
            workers = build_workers if build_workers and build_workers > 0 else max(1, os.cpu_count() or 4)
            png_index = self._build_png_blob(uniq, blob_path, compress_level, workers, read_chunk)
            np.save(idx_path, png_index)
            meta_path.write_text(json.dumps(want), encoding="utf-8")

        local_rows = np.searchsorted(uniq, frame_ids).astype(np.int64)
        return WindowCache(
            window_frame_rows=torch.from_numpy(local_rows),
            window_actions=self._window_actions(windows, seq_len),  # kept on CPU; moved per-batch
            seq_len=seq_len,
            compute_device=compute_device,
            backing="png",
            disk_path=blob_path,
            png_blob_path=blob_path,
            png_index=png_index,
            png_frame_shape=(channels, height, width),
        )

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


@dataclass(eq=False)
class WindowCache:
    """Every sequence window for one split, decoded once and kept resident.

    ``frames_u8`` holds only the *unique* pixel frames (``uint8``) referenced by
    the windows, on ``frames_device`` (GPU when it fits, else CPU RAM).
    ``window_frame_rows[i]`` gathers window ``i``'s ``seq_len`` frames out of it.
    ``take()`` produces a model-ready, float-normalized batch on
    ``compute_device`` with zero disk access.

    ``backing == "png"`` is the exception: there is no ``frames_u8`` tensor --
    frames live as independently-seekable PNG blobs (``png_blob_path`` +
    ``png_index``, see ``PushTHDF5Sampler.materialize_windows_png``), and
    ``png_frame_shape`` stands in for ``frames_u8.shape[1:]``. This tier always
    goes through ``gather_cpu`` + ``finalize`` (never ``take()``), same as memmap.
    """

    window_frame_rows: torch.Tensor  # [N, seq_len] int64, indices into frames_u8 / png_index
    window_actions: torch.Tensor     # [N, seq_len, frameskip*A] float32
    seq_len: int
    compute_device: torch.device
    frames_u8: torch.Tensor | None = None  # [F, C, H, W] uint8; unused when backing == "png"
    backing: str = "memory"          # "cuda" | "cpu" | "memmap" | "png"
    disk_path: Path | None = None
    png_blob_path: Path | None = None       # concatenated PNG bytes, one run per unique frame
    png_index: np.ndarray | None = None     # [F, 2] int64 (offset, length) into png_blob_path
    png_frame_shape: tuple[int, int, int] | None = None  # (C, H, W), since there is no frames_u8

    def __post_init__(self) -> None:
        self._mean = IMAGE_MEAN.to(self.compute_device, dtype=torch.float32)
        self._std = IMAGE_STD.to(self.compute_device, dtype=torch.float32)
        self._frame_shape = tuple(self.frames_u8.shape[1:]) if self.frames_u8 is not None else self.png_frame_shape

    def __len__(self) -> int:
        return self.window_frame_rows.shape[0]

    def nbytes(self) -> int:
        if self.frames_u8 is not None:
            frames_bytes = self.frames_u8.element_size() * self.frames_u8.nelement()
        elif self.png_blob_path is not None and self.png_blob_path.exists():
            frames_bytes = self.png_blob_path.stat().st_size  # actual compressed size on disk
        else:
            frames_bytes = 0
        return (
            frames_bytes
            + self.window_frame_rows.element_size() * self.window_frame_rows.nelement()
            + self.window_actions.element_size() * self.window_actions.nelement()
        )

    def _gather_png(self, rows: torch.Tensor) -> torch.Tensor:
        """Seek + decode PNG bytes for the requested unique-frame rows. Opens
        its own file handle per call so concurrent prefetch threads (each
        running its own ``gather_cpu``) never share a seek position."""

        Image = require_pillow()
        rows_np = rows.numpy()
        c, h, w = self._frame_shape
        out = np.empty((rows_np.shape[0], h, w, c), dtype=np.uint8)
        with open(self.png_blob_path, "rb") as fh:
            for i, r in enumerate(rows_np):
                offset, length = self.png_index[r]
                fh.seek(int(offset))
                out[i] = np.asarray(Image.open(io.BytesIO(fh.read(int(length)))).convert("RGB"))
        chw = np.ascontiguousarray(np.moveaxis(out, 3, 1))  # [n, H, W, C] -> [n, C, H, W]
        return torch.from_numpy(chw)

    def gather_cpu(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """CPU-only, thread-safe: gather raw ``uint8`` pixels + actions for a
        batch (page-faulting / NVMe read for memmap; seek+PNG-decode for png).
        Pins the result when possible so the follow-up H2D copy can overlap."""

        idx = idx.to("cpu", dtype=torch.long)
        rows = self.window_frame_rows.index_select(0, idx).reshape(-1)
        if self.backing == "png":
            pixels_u8 = self._gather_png(rows)
        else:
            pixels_u8 = self.frames_u8.index_select(0, rows)      # [batch*seq_len, C, H, W] uint8
        action = self.window_actions.index_select(0, idx)
        try:
            pixels_u8 = pixels_u8.pin_memory()
            action = action.pin_memory()
        except Exception:  # noqa: BLE001 - no CUDA / pinning unavailable
            pass
        return {"pixels_u8": pixels_u8, "action": action, "batch": idx.shape[0]}

    def finalize(self, raw: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        """Move a ``gather_cpu`` result to ``device`` and float-normalize (cheap;
        runs on the consumer thread)."""

        pixels = raw["pixels_u8"].to(device, non_blocking=True)
        pixels = pixels.reshape(raw["batch"], self.seq_len, *self._frame_shape)
        pixels = pixels.float().div_(255.0).sub_(self._mean).div_(self._std)
        return {"pixels": pixels, "action": raw["action"].to(device, non_blocking=True)}

    def take(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """A ready ``{'pixels', 'action'}`` batch on ``compute_device`` (used by
        the GPU/RAM tiers and by eval; the memmap tier goes through
        ``gather_cpu`` + ``finalize`` on separate threads)."""

        idx = idx.to(self.window_frame_rows.device, dtype=torch.long)
        batch = idx.shape[0]
        rows = self.window_frame_rows.index_select(0, idx).reshape(-1)
        frames = self.frames_u8.index_select(0, rows)
        pixels = frames.to(self.compute_device, non_blocking=True).reshape(batch, self.seq_len, *self._frame_shape)
        pixels = pixels.float().div_(255.0).sub_(self._mean).div_(self._std)
        action = self.window_actions.index_select(0, idx.to(self.window_actions.device))
        return {"pixels": pixels, "action": action.to(self.compute_device, non_blocking=True)}


def _window_index_stream(n: int, batch_size: int, seed: int):
    """Infinite stream of shuffled window-index tensors (one per minibatch)."""

    epoch = 0
    while True:
        order = np.random.default_rng(seed + epoch).permutation(n)
        for start in range(0, n, batch_size):
            yield torch.from_numpy(order[start : start + batch_size])
        epoch += 1


def windowcache_batch_stream(
    cache: WindowCache, batch_size: int, seed: int, prefetch: int = 0, workers: int = 1
):
    """Minibatch stream over a WindowCache.

    ``prefetch <= 0``: synchronous ``take`` (GPU/RAM tiers - no transfer to hide).
    ``prefetch > 0``: a thread pool runs the ``gather_cpu`` reads ``prefetch``
    batches ahead (ordered) while the consumer thread does the small H2D +
    normalize - this is what hides NVMe latency for the memmap tier.
    """

    idxs = _window_index_stream(len(cache), batch_size, seed)
    if prefetch <= 0:
        for idx in idxs:
            yield cache.take(idx)
        return

    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="wcache")
    try:
        inflight = deque(pool.submit(cache.gather_cpu, next(idxs)) for _ in range(prefetch))
        while True:
            done = inflight.popleft()
            inflight.append(pool.submit(cache.gather_cpu, next(idxs)))
            yield cache.finalize(done.result(), cache.compute_device)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def build_eval_batches(
    sampler: PushTHDF5Sampler,
    split: str,
    window_ids: np.ndarray,
    batch_size: int,
    seq_len: int,
    count: int | None,
    seed: int,
    compute_device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Fixed list of eval minibatches, decoded once (GPU-resident, small)."""

    ids = np.asarray(window_ids, dtype=np.int64)
    if count is not None:
        keep = min(len(ids), count * batch_size)
        ids = np.random.default_rng(seed).choice(ids, size=keep, replace=False)
    cache = sampler.materialize_windows(split, seq_len, ids, compute_device, compute_device)
    return [
        cache.take(torch.arange(start, min(start + batch_size, len(cache))))
        for start in range(0, len(cache), batch_size)
    ]


def resolve_cache_placement(
    mode: str,
    est_frames: int,
    pixel_shape: tuple[int, ...],
    compute_device: torch.device,
    gpu_budget_gb: float,
    ram_budget_gb: float,
    disk_budget_gb: float,
    memmap_dir: Path,
    gpu_reserve_gb: float = 14.0,
    existing_cache_gb: dict[str, float] | None = None,
) -> tuple[bool, torch.device, str, str]:
    """Decide where the resident training cache lives.

    Returns ``(use_cache, frames_device, backing, reason)`` where ``backing`` is
    ``"cuda"`` (GPU), ``"cpu"`` (host RAM), ``"memmap"`` (on-disk NVMe, raw
    uint8, page cached), ``"png"`` (on-disk, lossless PNG blob, ~13x smaller
    than raw -- for when even the raw memmap doesn't fit the disk budget) or
    ``"none"`` (fall back to the per-batch HDF5 stream).

    ``auto`` tries, in order: GPU -> host RAM -> disk memmap (raw) -> disk PNG
    (compressed) -> streaming. PNG is strictly a fallback for when raw memmap
    doesn't fit -- decoding it costs a bit of CPU per batch, so memmap is
    preferred whenever both fit.

    ``existing_cache_gb`` (from ``PushTHDF5Sampler.existing_cache_gb``) short-
    circuits the disk-space check for a tier whose file is ALREADY built and
    matches this exact selection: reusing it needs zero new bytes, so it must
    not be rejected by a free-disk-space estimate sized for a fresh REBUILD --
    that estimate naturally shrinks once the existing cache itself has eaten
    into free disk space, which used to make an already-finished cache get
    rejected right after it finished building.

    A GPU-resident cache competes with the model + activations + (for MCSDCA)
    the retained-sample chain, so it is only chosen when ``est_gb`` fits under
    both ``gpu_budget_gb`` and ``free - gpu_reserve_gb`` (headroom left for
    training). Otherwise it drops to host RAM / disk, which is nearly free once
    prefetch hides the transfer.
    """

    existing_cache_gb = existing_cache_gb or {}

    _, height, width, channels = pixel_shape
    est_gb = est_frames * height * width * channels / 1e9
    cpu = torch.device("cpu")
    if mode == "off":
        return False, cpu, "none", f"disabled (--window-cache off); would need ~{est_gb:.1f} GB"

    if mode in ("auto", "gpu") and compute_device.type == "cuda":
        free_gb = float("inf")
        try:
            free_bytes, _ = torch.cuda.mem_get_info(compute_device)
            free_gb = free_bytes / 1e9
        except Exception:  # noqa: BLE001 - fall through to the numeric budget
            pass
        reserve = 0.0 if mode == "gpu" else max(0.0, gpu_reserve_gb)  # explicit --window-cache gpu skips the reserve
        gpu_cap = min(gpu_budget_gb, max(0.0, free_gb - reserve))
        if est_gb <= gpu_cap:
            return True, compute_device, "cuda", (
                f"GPU-resident (~{est_gb:.1f} GB <= {gpu_cap:.1f} GB; "
                f"{free_gb:.0f} GB free, {reserve:.0f} GB reserved for training)"
            )
        if mode == "gpu":
            return False, cpu, "none", f"forced GPU cache too big (~{est_gb:.1f} GB > {gpu_cap:.1f} GB free)"

    if mode in ("auto", "cpu"):
        # ram_budget_gb <= 0 means "auto": scale with the RAM this box actually
        # has free. A positive value is an explicit ceiling, still clamped to
        # what is free. Slab-wise build peaks at ~resident + one read chunk, so
        # the safety margin is small.
        avail_gb = None
        try:
            import psutil

            avail_gb = psutil.virtual_memory().available / 1e9
        except Exception:  # noqa: BLE001 - psutil optional
            pass
        if ram_budget_gb and ram_budget_gb > 0:
            ram_cap = min(ram_budget_gb, 0.85 * avail_gb) if avail_gb is not None else ram_budget_gb
        else:
            ram_cap = 0.6 * avail_gb if avail_gb is not None else 64.0
        if est_gb * 1.05 <= ram_cap:
            note = f", {avail_gb:.0f} GB free" if avail_gb is not None else ""
            return True, cpu, "cpu", f"CPU-RAM-resident (~{est_gb:.1f} GB <= {ram_cap:.1f} GB cap{note})"
        if mode == "cpu":
            return False, cpu, "none", (
                f"streaming fallback (~{est_gb:.1f} GB > {ram_cap:.1f} GB RAM cap; "
                f"try cache.mode=memmap or a smaller data.fractions)"
            )

    free_disk_gb = float("inf")
    if mode in ("auto", "memmap", "png"):
        try:
            free_disk_gb = shutil.disk_usage(str(memmap_dir)).free / 1e9
        except Exception:  # noqa: BLE001 - directory may not exist yet
            try:
                free_disk_gb = shutil.disk_usage(str(Path(memmap_dir).anchor or ".")).free / 1e9
            except Exception:  # noqa: BLE001
                pass
        disk_cap = min(disk_budget_gb, 0.9 * free_disk_gb)

    if mode in ("auto", "memmap"):
        if "memmap" in existing_cache_gb:
            return True, cpu, "memmap", (
                f"disk memmap: found existing cache (~{existing_cache_gb['memmap']:.1f} GB in "
                f"{memmap_dir}), reuse"
            )
        if est_gb <= disk_cap:
            return True, cpu, "memmap", (
                f"disk memmap (~{est_gb:.1f} GB in {memmap_dir}, page-cached; "
                f"{free_disk_gb:.0f} GB free)"
            )
        if mode == "memmap":
            return False, cpu, "none", (
                f"forced disk memmap too big (~{est_gb:.1f} GB > {disk_cap:.1f} GB); "
                f"try cache.mode=png or raise cache.disk_gb"
            )
        # mode == "auto": raw memmap doesn't fit -- fall through and try the
        # compressed PNG blob before giving up to unprefetched streaming.

    if mode in ("auto", "png"):
        if "png" in existing_cache_gb:
            return True, cpu, "png", (
                f"disk PNG cache: found existing cache (~{existing_cache_gb['png']:.1f} GB in "
                f"{memmap_dir}), reuse"
            )
        est_png_gb = est_gb / PNG_COMPRESSION_RATIO_ESTIMATE
        if est_png_gb <= disk_cap:
            return True, cpu, "png", (
                f"disk PNG cache (~{est_png_gb:.1f} GB compressed, lossless, from ~{est_gb:.1f} GB raw, "
                f"in {memmap_dir}; {free_disk_gb:.0f} GB free)"
            )
        if mode == "png":
            return False, cpu, "none", (
                f"forced PNG cache too big (~{est_png_gb:.1f} GB > {disk_cap:.1f} GB); raise cache.disk_gb"
            )
        return False, cpu, "none", (
            f"streaming fallback (~{est_gb:.1f} GB raw / ~{est_png_gb:.1f} GB PNG > {disk_cap:.1f} GB disk cap; "
            f"raise cache.ram_gb / cache.disk_gb or use a smaller data.fractions)"
        )

    return False, cpu, "none", f"streaming (mode={mode}, device={compute_device.type})"


def batch_for_model(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


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
        **_predictor_collapse(pred, target, prefix="roll_"),
    }


# --------------------------------------------------------------------------- #
# Representation-collapse diagnostics (encoder side vs predictor side)         #
#                                                                             #
# The LeWM objective has NO stop-gradient on the target, so SIGReg alone      #
# resists collapse. These indicators localise a collapse:                     #
#   * enc_* low  (emb variance/std/norm tiny, many dead dims) => ENCODER      #
#     representation itself has shrunk.                                        #
#   * enc_* healthy but pred_target_var_ratio << 1 => the PREDICTOR output    #
#     collapsed to a near-constant while the target is still expressive.      #
#   * both low => encoder collapse dragging the predictor with it.            #
# --------------------------------------------------------------------------- #
def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]).float()


def _predictor_collapse(pred: torch.Tensor, target: torch.Tensor, prefix: str = "") -> dict[str, float]:
    p, t = _flat(pred), _flat(target)
    pv = float(p.var(dim=0, unbiased=False).mean())
    tv = float(t.var(dim=0, unbiased=False).mean())
    pn = float(p.norm(dim=-1).mean())
    tn = float(t.norm(dim=-1).mean())
    # Ratio ~1 = healthy; << 1 = the predictor output collapsed relative to the
    # (still expressive) target. Capped at 10 so the "< 1" regime stays readable
    # even when the target itself has collapsed (then trust enc_* instead).
    return {
        f"{prefix}pred_var_mean": pv,
        f"{prefix}target_var_mean": tv,
        f"{prefix}pred_target_var_ratio": min(pv / tv, 10.0) if tv > 1e-9 else 10.0,
        f"{prefix}pred_norm_mean": pn,
        f"{prefix}pred_target_norm_ratio": min(pn / tn, 10.0) if tn > 1e-9 else 10.0,
    }


@torch.no_grad()
def collapse_stats(pred: torch.Tensor, target: torch.Tensor, emb: torch.Tensor) -> dict[str, float]:
    """Encoder- + predictor-side collapse indicators for one one-step batch."""

    e = _flat(emb)
    ev = e.var(dim=0, unbiased=False)  # (D,)
    return {
        "enc_emb_var_mean": float(ev.mean()),
        "enc_emb_std_mean": float(ev.clamp_min(0.0).sqrt().mean()),
        "enc_emb_norm_mean": float(e.norm(dim=-1).mean()),
        "enc_dead_dim_frac": float((ev < 1e-4).float().mean()),
        **_predictor_collapse(pred, target),
    }


@torch.no_grad()
def average_collapse(
    model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], history_size: int, num_preds: int
) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for batch in batches:
        pred, target, emb = predict_target(model, batch_for_model(model, batch), history_size, num_preds)
        for key, value in collapse_stats(pred, target, emb).items():
            acc.setdefault(key, []).append(value)
    return {f"col_{key}": float(np.mean(vals)) for key, vals in acc.items()}


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
            carry = ("latent_norm_drift", "target_latent_norm", "pred_latent_variance",
                     "roll_pred_target_var_ratio", "roll_pred_target_norm_ratio")
            for key in carry:
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
    collapse = average_collapse(model, val_batches, profile.history_size, profile.num_preds)
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
        **collapse,
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
class EarlyStopper:
    """MCSDCA-paper early stopping: halt when the validation loss has not improved
    (by more than ``min_delta``) for ``patience`` consecutive periodic evals. With
    ``eval_every_epochs = 1`` one eval == one epoch, matching the paper's
    "4 consecutive epochs" rule."""

    def __init__(self, enabled: bool, patience: int, min_delta: float) -> None:
        self.enabled = bool(enabled)
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.best = float("inf")
        self.bad = 0

    @classmethod
    def from_args(cls, args: Any) -> "EarlyStopper":
        return cls(
            getattr(args, "early_stop_enabled", False),
            getattr(args, "early_stop_patience", 4),
            getattr(args, "early_stop_min_delta", 0.0),
        )

    def should_stop(self, val_mse: float) -> bool:
        if not self.enabled:
            return False
        if val_mse < self.best - self.min_delta:
            self.best = val_mse
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience


def train_baseline(
    name: str,
    model: torch.nn.Module,
    make_stream: Callable[[int], Any],
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
    stream = make_stream(args.seed)
    stopper = EarlyStopper.from_args(args)
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
        periodic = step % eval_interval == 0
        if step == 1 or step == budget or periodic:
            row = evaluate(
                name, model, train_batches, val_batches, profile, step, train_time_s,
                {"learning_rate": scheduler.get_last_lr()[0], "status": "ok"},
            )
            rows.append(row)
            progress.set_postfix(train=f"{row['train_mse']:.4g}", val=f"{row['val_mse']:.4g}")
            if periodic and step != budget and stopper.should_stop(float(row["val_mse"])):
                row["early_stopped"] = True
                tqdm.write(f"[{name}] early stop @ backprop {step} "
                           f"(val_mse no improve for {stopper.patience} evals; best={stopper.best:.4g})")
                break
    progress.close()
    return rows


def train_mcsdca(
    name: str,
    model: torch.nn.Module,
    make_stream: Callable[[int], Any],
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
    stream = make_stream(args.seed)
    last_batch: dict[str, dict[str, torch.Tensor]] = {}

    def loss_fn() -> torch.Tensor:
        batch = batch_for_model(model, next(stream))  # fresh minibatch per inner Langevin step
        last_batch["value"] = batch
        return training_objective(model, batch, profile, sigreg, args.sigreg_weight)

    stopper = EarlyStopper.from_args(args)
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
        periodic = bc >= next_eval_at
        if optimizer.outer_step == 1 or bc >= budget or periodic:
            row = evaluate(
                name, model, train_batches, val_batches, profile, bc, train_time_s,
                {**mcsdca_extra(info), "status": "ok"},
            )
            rows.append(row)
            progress.set_postfix(outer=info["outer_step"], s_loss=f"{info['loss']:.4g}", val=f"{row['val_mse']:.4g}")
            if periodic:
                next_eval_at = ((bc // eval_interval) + 1) * eval_interval
                if bc < budget and stopper.should_stop(float(row["val_mse"])):
                    row["early_stopped"] = True
                    tqdm.write(f"[{name}] early stop @ backprop {bc} "
                               f"(val_mse no improve for {stopper.patience} evals; best={stopper.best:.4g})")
                    break
    progress.close()
    return rows


# --------------------------------------------------------------------------- #
# Result IO                                                                    #
# --------------------------------------------------------------------------- #
def save_checkpoint(run_dir: Path, model: torch.nn.Module) -> Path:
    path = run_dir / "checkpoints" / "full_model.pt"
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
# In-process reuse across sweep points                                         #
#                                                                             #
# ``sweep.run_grid`` calls ``run()`` once per grid point in the same process. #
# Only the optimizer hyper-parameters change between most points, so the      #
# sampler, the resident window cache and the seed-specific initial weights    #
# are memoized and shared instead of rebuilt (and re-read from disk) each     #
# time. Keys capture everything an entry depends on; small LRU bounds keep    #
# peak memory sane when the odLD sweep varies ``seed``.                       #
# --------------------------------------------------------------------------- #
_SAMPLER_MEMO: "OrderedDict[tuple, PushTHDF5Sampler]" = OrderedDict()
_CACHE_MEMO: "OrderedDict[tuple, WindowCache]" = OrderedDict()
_MODEL_MEMO: "OrderedDict[tuple, tuple[torch.nn.Module, dict[str, Any]]]" = OrderedDict()
_BASESTATE_MEMO: "OrderedDict[tuple, dict[str, torch.Tensor]]" = OrderedDict()


def _memoize(memo: "OrderedDict[tuple, Any]", key: tuple, factory: Callable[[], Any], maxsize: int) -> Any:
    if key in memo:
        memo.move_to_end(key)
        return memo[key]
    value = factory()
    memo[key] = value
    memo.move_to_end(key)
    while len(memo) > max(1, maxsize):
        _, evicted = memo.popitem(last=False)
        closer = getattr(evicted, "close", None)
        if callable(closer):
            closer()
        del evicted
        gc.collect()
    return value


def get_sampler(
    data_path: Path, frameskip: int, max_seq_len: int, action_stats_samples: int,
    seed: int, train_fraction: float,
) -> "tuple[PushTHDF5Sampler, tuple]":
    key = ("sampler", str(data_path), frameskip, max_seq_len, action_stats_samples, seed, round(train_fraction, 9))
    sampler = _memoize(
        _SAMPLER_MEMO, key,
        lambda: PushTHDF5Sampler(data_path, frameskip, max_seq_len, action_stats_samples, seed, train_fraction),
        maxsize=3,
    )
    return sampler, key


def _model_key(model_cfg: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(model_cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def get_model(model_cfg: dict[str, Any], device: torch.device) -> "tuple[torch.nn.Module, dict[str, Any]]":
    key = ("model", _model_key(model_cfg), str(device))
    return _memoize(_MODEL_MEMO, key, lambda: initialize_lewm(model_cfg, device), maxsize=1)


def get_base_state(model_cfg: dict[str, Any], seed: int) -> dict[str, torch.Tensor]:
    key = ("base_state", _model_key(model_cfg), seed)

    def factory() -> dict[str, torch.Tensor]:
        seed_everything(seed)  # reproduce "seed then instantiate" so init stays seed-specific
        tmp = instantiate(OmegaConf.create(model_cfg)).to("cpu")
        enable_full_model_training(tmp)
        state = clone_cpu_state(tmp)
        del tmp
        gc.collect()
        return state

    return _memoize(_BASESTATE_MEMO, key, factory, maxsize=4)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run(args: SimpleNamespace, output_dir: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Train every optimizer in ``args.optimizers`` for one (data_fraction, seed).

    Each optimizer writes into its own leaf folder so re-running a single
    algorithm (e.g. after tweaking its hyperparameters, same seed) never touches
    the others' results:

        output_dir/<optimizer_key>/<run_id>/
            config.yaml  metrics.csv  run.json
            checkpoints/full_model.pt   (when ``args.save_checkpoints``)

    ``run_id`` is ``<stamp>__seed<seed>``, plus ``__<run_tag>`` when
    ``args.run_tag`` is set -- a human-readable label to tell repeated runs at
    the same seed apart without opening ``run.json``. A fresh timestamp every
    call means reruns never overwrite an earlier one.

    Returns ``(run_dirs, per_optimizer_final_rows)`` where ``run_dirs`` maps
    optimizer name -> its leaf folder.
    """

    require_hdf5()
    data_path = Path(args.data_path).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_tag = str(getattr(args, "run_tag", "") or "").strip()
    run_id = f"{stamp}__seed{args.seed}" + (f"__{run_tag}" if run_tag else "")
    output_dir = Path(output_dir).resolve()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tune_compute(device, getattr(args, "num_threads", 0))
    seed_everything(args.seed)

    profile = profile_from_args(args)
    training_seq_len = profile.history_size + profile.num_preds
    max_seq_len = profile.history_size + max(ROLLOUT_HORIZONS)

    sampler, sampler_key = get_sampler(
        data_path, profile.frameskip, max_seq_len, args.action_stats_samples, args.seed, profile.train_fraction
    )
    model, model_config = get_model(args.model, device)
    base_state = get_base_state(args.model, args.seed)

    train_window_ids = sampler.select_window_ids("train", training_seq_len, profile.data_fraction, args.seed + 2)
    val_window_ids = sampler.select_window_ids("val", max_seq_len, profile.val_eval_fraction, args.seed + 7)
    steps_per_epoch = math.ceil(len(train_window_ids) / profile.batch_size)
    budget = profile.default_budget or steps_per_epoch * profile.epochs
    if args.eval_interval:
        eval_interval = args.eval_interval
    elif getattr(args, "eval_every_epochs", None):
        eval_interval = max(1, int(args.eval_every_epochs) * steps_per_epoch)
    else:
        eval_interval = max(1, math.ceil(budget * args.eval_interval_frac))

    # ---- resident training-window cache (plans 1-4): decode once, reuse ---- #
    memmap_dir = Path(args.memmap_dir).resolve() if args.memmap_dir else (data_path.parent / ".framecache")
    est_frames = sampler.count_unique_frames("train", training_seq_len, train_window_ids)
    existing = sampler.existing_cache_gb("train", training_seq_len, train_window_ids, memmap_dir)
    use_cache, frames_device, backing, cache_reason = resolve_cache_placement(
        args.window_cache, est_frames, sampler.pixel_shape, device,
        args.cache_max_gb, args.cache_ram_gb, args.cache_disk_gb, memmap_dir,
        gpu_reserve_gb=getattr(args, "cache_gpu_reserve_gb", 14.0),
        existing_cache_gb=existing,
    )
    cache_bytes = 0
    prefetch = 0
    if use_cache:
        cache_key = (
            "cache", sampler_key, "train", training_seq_len,
            round(profile.data_fraction, 12), args.seed + 2, backing, str(device),
        )
        if backing == "memmap":
            builder: Callable[[], WindowCache] = lambda: sampler.materialize_windows_memmap(
                "train", training_seq_len, train_window_ids, memmap_dir, device
            )
        elif backing == "png":
            builder = lambda: sampler.materialize_windows_png(
                "train", training_seq_len, train_window_ids, memmap_dir, device,
                compress_level=getattr(args, "png_compress_level", 6),
                build_workers=getattr(args, "png_build_workers", 0),
            )
        else:
            builder = lambda: sampler.materialize_windows(
                "train", training_seq_len, train_window_ids, frames_device, device
            )
        train_cache = _memoize(_CACHE_MEMO, cache_key, builder, maxsize=max(1, args.cache_memo_size))
        cache_bytes = train_cache.nbytes()
        prefetch = 0 if backing == "cuda" else max(0, args.prefetch_depth)

        def make_stream(seed: int):
            return windowcache_batch_stream(
                train_cache, profile.batch_size, seed,
                prefetch=prefetch, workers=max(1, args.prefetch_workers),
            )
    else:
        def make_stream(seed: int):
            return training_batch_stream(
                sampler, train_window_ids, profile.batch_size, training_seq_len, seed
            )

    train_batches = build_eval_batches(
        sampler, "train", train_window_ids, profile.eval_batch_size, training_seq_len,
        profile.eval_train_batches, args.seed + 3, device,
    )
    val_batches = build_eval_batches(
        sampler, "val", val_window_ids, profile.eval_batch_size, max_seq_len,
        profile.val_batches, args.seed + 4, device,
    )

    try:
        from stable_worldmodel.wm.loss import SIGReg
    except ImportError as exc:
        raise RuntimeError("Full LeWM training requires stable-worldmodel with SIGReg.") from exc
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=profile.sigreg_num_proj).to(device)

    mcsdca_config = args.mcsdca_config
    optimizers = parse_optimizers(args.optimizers)

    print(
        f"data_fraction={profile.data_fraction:.2%} "
        f"train_windows={len(train_window_ids):,} budget={budget:,} "
        f"(~{budget / steps_per_epoch:.1f} epochs) eval_interval={eval_interval:,}"
    )
    if use_cache:
        where = "on disk" if backing in ("memmap", "png") else "resident"
        print(f"  window-cache: {cache_reason}  [{cache_bytes / 1e9:.1f} GB {where}, "
              f"{est_frames:,} unique frames, prefetch={prefetch}]")
    else:
        print(f"  window-cache: {cache_reason}")

    def payload(name: str, final: dict[str, Any]) -> dict[str, Any]:
        args_dump = {k: v for k, v in vars(args).items() if k not in ("model", "mcsdca_config", "config_yaml")}
        return {
            "args": args_dump,
            "device": str(device),
            "run_id": run_id,
            "optimizer": name,
            "dataset": sampler.describe(),
            "model_config": model_config,
            "profile": {"name": "yaml", **asdict(profile)},
            "mcsdca_config": asdict(mcsdca_config),
            "budget": int(budget),
            "steps_per_epoch": int(steps_per_epoch),
            "epochs_equivalent": budget / steps_per_epoch,
            "train_windows": int(len(train_window_ids)),
            "val_windows": int(len(val_window_ids)),
            "window_cache": {
                "used": bool(use_cache),
                "reason": cache_reason,
                "backing": backing,
                "frames_device": str(frames_device) if use_cache else None,
                "est_unique_frames": int(est_frames),
                "bytes": int(cache_bytes),
                "prefetch": int(prefetch),
            },
            "final_result": final,
        }

    run_dirs: dict[str, Path] = {}
    final_results: list[dict[str, Any]] = []

    for name in optimizers:
        opt_dir = output_dir / optimizer_key(name) / run_id
        opt_dir.mkdir(parents=True, exist_ok=True)
        if getattr(args, "config_yaml", None):
            (opt_dir / "config.yaml").write_text(args.config_yaml, encoding="utf-8")

        tqdm.write(f"[{name}] training")
        seed_everything(args.seed)
        restore_state(model, base_state, device)
        try:
            if name in BASELINE_OPTIMIZERS:
                rows = train_baseline(
                    name, model, make_stream, train_batches, val_batches,
                    profile, sigreg, args, budget, eval_interval,
                )
            else:
                rows = train_mcsdca(
                    name, model, make_stream, train_batches, val_batches,
                    profile, sigreg, mcsdca_config, args, budget, eval_interval,
                )
            final = dict(rows[-1])
        except Exception as exc:  # noqa: BLE001 - a single optimizer must not kill the sweep
            traceback.print_exc()
            final = {"optimizer": name, "status": "diverged", "error": repr(exc)}
            rows = [final]
        final_results.append(final)
        if args.save_checkpoints and final.get("status") == "ok":
            save_checkpoint(opt_dir, model)
        write_metrics(opt_dir, rows)
        write_run_json(opt_dir, payload(name, final))
        run_dirs[name] = opt_dir
        if final.get("status") == "ok":
            tqdm.write(f"[{name}] ok val_mse={final['val_mse']:.6g} backprop_calls={final['backprop_calls']}  -> {opt_dir}")
        else:
            tqdm.write(f"[{name}] diverged: {final.get('error')}  -> {opt_dir}")

    return run_dirs, final_results


DEFAULT_CONFIG = ROOT / "configs" / "experiment1.yaml"
OUTPUT_ROOT = ROOT / "outputs"  # runs land in outputs/<dataTAG>/<optimizer>/<stamp>__seed<seed>[__tag]/


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> Any:
    """Load the single experiment YAML and apply ``key=value`` dotlist overrides."""

    cfg = OmegaConf.load(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def data_tag(fraction: float) -> str:
    """``0.04 -> data4``, ``0.001 -> data0p1`` (percent, '.' -> 'p')."""

    return "data" + ("%g" % (fraction * 100.0)).replace(".", "p")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeWM PushT predictor optimizer: AdamW vs MCSDCA (single-YAML, no tuning).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the experiment YAML.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Dotlist override, repeatable (e.g. --set train.budget=400).")
    parser.add_argument("--device", default=None, help="Override cfg.device (auto, cpu, cuda, cuda:N).")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.device:
        cfg.device = args.device
    fraction = float(cfg.data.fractions[0])
    seed = int(cfg.seed[0])
    out_dir = OUTPUT_ROOT / data_tag(fraction)
    run_args = build_run_args(cfg, data_fraction=fraction, seed=seed)
    print(f"[engine] one run: {data_tag(fraction)} seed={seed} optimizers={list(cfg.optimizers)}")
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return
    run_dirs, _ = run(run_args, out_dir)
    for name, opt_dir in run_dirs.items():
        print(f"[engine] -> [{name}] {opt_dir}")


if __name__ == "__main__":
    main()
