"""Grid sweep runner for ``run_pusht_predictor_experiment.run()``.

Reads a YAML grid file, runs every point of the Cartesian product in-process,
and appends one row per (config, optimizer) to ``<output>/sweep_summary.csv``.
Re-running skips points whose ``run_id`` is already in the CSV.

Selection rule (``select_best``):
    primary  = min val rollout MSE@5
    tiebreak = min val one-step MSE
    rejected = status != ok; non-finite / blown-up val_mse (>= 10); latent
               collapse (pred_latent_variance < 1e-5); or |latent_norm_drift|
               > target_latent_norm (rollout latent scale ran away).

YAML schema::

    base:   {arg: value}          # applied to every run (argparse dests)
    grid:   {arg: [values, ...]}  # swept; Cartesian product
    output: outputs/sweep/<name>  # run dirs + sweep_summary.csv land here
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_pusht_predictor_experiment import build_parser, run

SUMMARY_METRICS = (
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


def slugify(combo: dict[str, Any]) -> str:
    return "__".join(f"{key}={value}" for key, value in combo.items()) or "base"


def load_grid(path: Path) -> tuple[dict, dict, Path]:
    spec = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    base = spec.get("base") or {}
    grid = spec.get("grid") or {}
    raw_output = Path(spec["output"])
    output_dir = raw_output if raw_output.is_absolute() else ROOT / raw_output
    return base, grid, output_dir


def combos(grid: dict[str, list]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


def make_args(base: dict, combo: dict, output_dir: Path) -> argparse.Namespace:
    args = build_parser().parse_args([])
    args.save_checkpoints = False
    args.output_dir = str(output_dir)
    for key, value in {**base, **combo}.items():
        if not hasattr(args, key):
            raise KeyError(f"Unknown argument '{key}' in grid file (not an argparse dest).")
        setattr(args, key, value)
    args.run_id = slugify(combo)
    return args


def done_run_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(encoding="utf-8") as handle:
        return {row["run_id"] for row in csv.DictReader(handle)}


def append_rows(summary_path: Path, rows: list[dict[str, Any]], grid_keys: list[str]) -> None:
    fieldnames = ["run_id", *grid_keys, "optimizer", *SUMMARY_METRICS, "error"]
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_grid(grid_path: Path) -> Path:
    base, grid, output_dir = load_grid(grid_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sweep_summary.csv"
    grid_keys = list(grid)
    already = done_run_ids(summary_path)
    points = combos(grid)
    print(f"{grid_path.name}: {len(points)} grid points -> {summary_path}")

    bar = tqdm(points, desc=grid_path.stem, unit="cfg", dynamic_ncols=True)
    for combo in bar:
        run_id = slugify(combo)
        bar.set_postfix_str(run_id[:60])
        if run_id in already:
            bar.write(f"skip {run_id} (already in summary)")
            continue
        final_results = run(make_args(base, combo, output_dir))
        rows = []
        for item in final_results:
            row: dict[str, Any] = {"run_id": run_id, **combo, "optimizer": item.get("optimizer")}
            for key in SUMMARY_METRICS:
                row[key] = item.get(key)
            row["error"] = item.get("error", "")
            rows.append(row)
        append_rows(summary_path, rows, grid_keys)
    print(f"wrote {summary_path}")
    return summary_path


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_best(summary_path: Path, optimizer: str | None = None) -> tuple[list[dict], dict | None]:
    with summary_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ranked: list[tuple[float, float, dict]] = []
    for row in rows:
        if optimizer and row.get("optimizer") != optimizer:
            continue
        if row.get("status") != "ok":
            continue
        val = _float(row.get("val_mse"))
        r5 = _float(row.get("rollout_mse_5"))
        drift = _float(row.get("latent_norm_drift"))
        tnorm = _float(row.get("target_latent_norm"))
        pvar = _float(row.get("pred_latent_variance"))
        if None in (val, r5, drift, tnorm, pvar):
            continue
        if not (val < 10.0 and pvar > 1e-5 and abs(drift) <= tnorm):
            continue
        ranked.append((r5, val, row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in ranked], (ranked[0][2] if ranked else None)


def print_ranking(summary_path: Path, optimizer: str | None) -> None:
    ranked, best = select_best(summary_path, optimizer)
    if not ranked:
        print("no configs passed the selection gates.")
        return
    grid_keys = [k for k in ranked[0] if k not in {"run_id", "optimizer", "error", *SUMMARY_METRICS}]
    for row in ranked[:10]:
        params = " ".join(f"{k}={row[k]}" for k in grid_keys)
        print(
            f"  rollout5={float(row['rollout_mse_5']):.5f} val={float(row['val_mse']):.5f} "
            f"gap={float(row['train_val_gap']):.5f} {row['optimizer']} {params}"
        )
    print(f"best: {best['run_id']} ({best['optimizer']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep over run_pusht_predictor_experiment.")
    parser.add_argument("grid", help="Path to a sweep grid YAML.")
    parser.add_argument("--select-only", action="store_true", help="Only rank an existing sweep_summary.csv.")
    parser.add_argument("--optimizer", default=None, help="Restrict ranking to one optimizer name.")
    args = parser.parse_args()

    grid_path = Path(args.grid).resolve()
    _, _, output_dir = load_grid(grid_path)
    summary_path = output_dir / "sweep_summary.csv"
    if not args.select_only:
        run_grid(grid_path)
    print_ranking(summary_path, args.optimizer)


if __name__ == "__main__":
    main()
