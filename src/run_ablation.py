"""Fixed-config PushT data-scaling ablation.

Runs a locked odLD / udLD / AdamW config at ``data_fraction`` in
``{0.1, 0.25, 0.5, 1.0}`` with a FIXED backprop budget at every fraction and one
or more seeds, then aggregates final metrics into
``outputs/ablation/<timestamp>/ablation.csv`` (one row per optimizer x fraction x
seed). ``epochs_equivalent = budget / steps_per_epoch(fraction)`` shows that
smaller fractions get more passes over less data - the regime where a
regularizing optimizer should help most.

    uv run python src/run_ablation.py configs/ablation.yaml
    uv run python src/run_ablation.py configs/ablation.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_pusht_predictor_experiment import build_parser, optimizer_key, run

ABLATION_METRICS = (
    "val_mse",
    "train_mse",
    "train_val_gap",
    "rollout_mse_1",
    "rollout_mse_3",
    "rollout_mse_5",
    "latent_norm_drift",
    "target_latent_norm",
    "pred_latent_variance",
    "backprop_calls",
    "status",
)


def make_args(
    profile: str,
    fraction: float,
    seed: int,
    budget: int,
    device: str,
    optimizer: str,
    overrides: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> argparse.Namespace:
    args = build_parser().parse_args([])
    args.save_checkpoints = False
    args.training_profile = profile
    args.data_fraction = fraction
    args.backprop_budget = budget
    args.seed = seed
    args.device = device
    args.optimizers = optimizer
    args.output_dir = str(output_dir)
    args.run_id = run_id
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise KeyError(f"Unknown override '{key}' for optimizer '{optimizer}'.")
        setattr(args, key, value)
    return args


def run_ablation(config_path: Path, cli: argparse.Namespace) -> Path:
    spec = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    optimizers: dict[str, dict] = spec.get("optimizers") or {}
    profile = "smoke" if cli.dry_run else spec.get("profile", "small")
    device = cli.device or spec.get("device", "auto")
    budget = cli.budget or (40 if cli.dry_run else int(spec["budget"]))
    fractions = cli.fractions or ([0.1, 1.0] if cli.dry_run else list(spec["fractions"]))
    seeds = cli.seeds or ([int(spec["seeds"][0])] if cli.dry_run else [int(s) for s in spec["seeds"]])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (ROOT / (cli.output or spec.get("output", "outputs/ablation"))) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation.csv"
    fieldnames = [
        "optimizer", "data_fraction", "seed", "epochs_equivalent", "steps_per_epoch",
        *ABLATION_METRICS, "error", "run_id",
    ]

    print(
        f"ablation: {list(optimizers)} x fractions={fractions} x seeds={seeds} "
        f"profile={profile} budget={budget} -> {csv_path}"
    )
    jobs = [
        (optimizer, overrides or {}, fraction, seed)
        for optimizer, overrides in optimizers.items()
        for fraction in fractions
        for seed in seeds
    ]
    rows: list[dict[str, Any]] = []
    bar = tqdm(jobs, desc="ablation", unit="run", dynamic_ncols=True)
    for optimizer, overrides, fraction, seed in bar:
        run_id = f"{optimizer_key(optimizer)}__frac{fraction}__seed{seed}"
        bar.set_postfix_str(run_id)
        args = make_args(
            profile, float(fraction), int(seed), int(budget), device,
            optimizer, overrides, output_dir, run_id,
        )
        final = run(args)[0]
        meta = json.loads((output_dir / run_id / "run.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "optimizer": optimizer,
            "data_fraction": fraction,
            "seed": seed,
            "epochs_equivalent": meta.get("epochs_equivalent"),
            "steps_per_epoch": meta.get("steps_per_epoch"),
            "run_id": run_id,
            "error": final.get("error", ""),
        }
        for key in ABLATION_METRICS:
            row[key] = final.get(key)
        rows.append(row)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="PushT data-scaling ablation with a fixed backprop budget.")
    parser.add_argument("config", help="Ablation YAML (optimizers + locked overrides, budget, fractions, seeds).")
    parser.add_argument("--dry-run", action="store_true", help="smoke profile, budget 40, fractions [0.1, 1.0], first seed.")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    cli = parser.parse_args()
    run_ablation(Path(cli.config).resolve(), cli)


if __name__ == "__main__":
    main()
