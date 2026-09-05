"""Grid sweep runner for ``run_pusht_predictor_experiment.run()``.

Reads a YAML grid file, runs every point of the Cartesian product in-process,
and appends one row per (config, optimizer) to ``<output>/sweep_summary.csv``.
Re-running skips points whose ``run_id`` is already in the CSV.

Selection rule (``select_best``):
    One config = all rows that share every grid parameter except ``seed``.
    primary  = min  seed-MEAN val rollout MSE@5
    tiebreak = min  seed-MEAN val one-step MSE
    rejected = any seed with status != ok; or the seed-MEAN fails a gate:
               non-finite / blown-up val_mse (>= 10); latent collapse
               (pred_latent_variance < 1e-5); or |latent_norm_drift| >
               target_latent_norm (rollout latent scale ran away).

YAML schema::

    base:    {arg: value}          # applied to every run (argparse dests)
    grid:    {arg: [values, ...]}  # swept; Cartesian product
    exclude: [{arg: value, ...}]   # optional; drop combos matching any entry
    output:  outputs/sweep/<name>  # run dirs + sweep_summary.csv land here

Special grid key ``mcsdca_epsilon_ratio``: not an argparse dest. When present,
each combo's ``mcsdca_epsilon`` is set to ``ratio * mcsdca_eta`` (eta taken from
the combo or ``base``), so the Langevin noise-to-signal regime is what varies
rather than an eta-dependent absolute.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
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


def load_grid(path: Path) -> tuple[dict, dict, list, Path]:
    spec = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    base = spec.get("base") or {}
    grid = spec.get("grid") or {}
    exclude = spec.get("exclude") or []
    raw_output = Path(spec["output"])
    output_dir = raw_output if raw_output.is_absolute() else ROOT / raw_output
    return base, grid, exclude, output_dir


def _values_match(left: Any, right: Any) -> bool:
    lf, rf = _float(left), _float(right)
    if lf is not None and rf is not None:
        return math.isclose(lf, rf, rel_tol=1e-9, abs_tol=0.0)
    return str(left) == str(right)


def combos(grid: dict[str, list], exclude: list[dict] | None = None) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    points = [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]
    if not exclude:
        return points

    def is_excluded(combo: dict[str, Any]) -> bool:
        return any(
            rule and all(key in combo and _values_match(combo[key], value) for key, value in rule.items())
            for rule in exclude
        )

    return [combo for combo in points if not is_excluded(combo)]


def make_args(base: dict, combo: dict, output_dir: Path) -> argparse.Namespace:
    args = build_parser().parse_args([])
    args.save_checkpoints = False
    args.output_dir = str(output_dir)
    merged = {**base, **combo}
    epsilon_ratio = merged.pop("mcsdca_epsilon_ratio", None)
    for key, value in merged.items():
        if not hasattr(args, key):
            raise KeyError(f"Unknown argument '{key}' in grid file (not an argparse dest).")
        setattr(args, key, value)
    if epsilon_ratio is not None:
        eta = merged.get("mcsdca_eta", args.mcsdca_eta)
        if eta is None:
            raise KeyError("mcsdca_epsilon_ratio needs mcsdca_eta (set it in base or grid).")
        args.mcsdca_epsilon = float(epsilon_ratio) * float(eta)
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
    base, grid, exclude, output_dir = load_grid(grid_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sweep_summary.csv"
    grid_keys = list(grid)
    already = done_run_ids(summary_path)
    points = combos(grid, exclude)
    dropped = f" ({len(combos(grid)) - len(points)} excluded)" if exclude else ""
    print(f"{grid_path.name}: {len(points)} grid points{dropped} -> {summary_path}")

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


def _group_key_columns(fieldnames: list[str]) -> list[str]:
    """Grid-parameter columns that identify one config: everything the sweep
    varied except the random ``seed``."""

    stop = fieldnames.index("optimizer") if "optimizer" in fieldnames else len(fieldnames)
    return [name for name in fieldnames[:stop] if name not in {"run_id", "seed"}]


def select_best(summary_path: Path, optimizer: str | None = None) -> tuple[list[dict], dict | None]:
    """Rank configs by seed-averaged val rollout MSE@5 (tiebreak: seed-averaged
    one-step val MSE). Rows sharing every grid parameter except ``seed`` are one
    config; it is rejected unless every seed ran ``status == ok`` and the
    seed-MEAN passes the finite / no-collapse / no-drift gates. The returned
    rows are copies of the first seed of each config plus ``n_seeds``,
    ``rollout_mse_5_mean`` and ``val_mse_mean``."""

    with summary_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        group_cols = _group_key_columns(list(reader.fieldnames or []))

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if optimizer and row.get("optimizer") != optimizer:
            continue
        groups.setdefault(tuple(row.get(col) for col in group_cols), []).append(row)

    ranked: list[tuple[float, float, dict]] = []
    for members in groups.values():
        if not members or any(member.get("status") != "ok" for member in members):
            continue
        series = {
            metric: [_float(member.get(metric)) for member in members]
            for metric in ("val_mse", "rollout_mse_5", "latent_norm_drift",
                           "target_latent_norm", "pred_latent_variance")
        }
        if any(value is None for values in series.values() for value in values):
            continue
        mean = {metric: sum(values) / len(values) for metric, values in series.items()}
        if not (
            mean["val_mse"] < 10.0
            and mean["pred_latent_variance"] > 1e-5
            and abs(mean["latent_norm_drift"]) <= mean["target_latent_norm"]
        ):
            continue
        representative = dict(members[0])
        representative["n_seeds"] = len(members)
        representative["rollout_mse_5_mean"] = mean["rollout_mse_5"]
        representative["val_mse_mean"] = mean["val_mse"]
        ranked.append((mean["rollout_mse_5"], mean["val_mse"], representative))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in ranked], (ranked[0][2] if ranked else None)


def print_ranking(summary_path: Path, optimizer: str | None) -> None:
    ranked, best = select_best(summary_path, optimizer)
    if not ranked:
        print("no configs passed the selection gates.")
        return
    reserved = {"run_id", "optimizer", "error", "seed", "n_seeds",
                "rollout_mse_5_mean", "val_mse_mean", *SUMMARY_METRICS}
    grid_keys = [k for k in ranked[0] if k not in reserved]
    for row in ranked[:10]:
        params = " ".join(f"{k}={row[k]}" for k in grid_keys)
        print(
            f"  rollout5_mean={float(row['rollout_mse_5_mean']):.5f} "
            f"val_mean={float(row['val_mse_mean']):.5f} "
            f"(n_seeds={row['n_seeds']}) {row['optimizer']} {params}"
        )
    print(f"best: {best['run_id']} ({best['optimizer']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep over run_pusht_predictor_experiment.")
    parser.add_argument("grid", help="Path to a sweep grid YAML.")
    parser.add_argument("--select-only", action="store_true", help="Only rank an existing sweep_summary.csv.")
    parser.add_argument("--optimizer", default=None, help="Restrict ranking to one optimizer name.")
    args = parser.parse_args()

    grid_path = Path(args.grid).resolve()
    *_, output_dir = load_grid(grid_path)
    summary_path = output_dir / "sweep_summary.csv"
    if not args.select_only:
        run_grid(grid_path)
    print_ranking(summary_path, args.optimizer)


if __name__ == "__main__":
    main()
