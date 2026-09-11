"""Experiment 1 driver: AdamW vs MCSDCA-odLD vs MCSDCA-udLD as a full-model LeWM
PushT predictor optimizer, faithful to the two source papers -- NO tuning.

Every knob lives in ``configs/experiment1.yaml``. This script sweeps
``data.fractions`` x ``seed``; each (fraction, seed) is one call to
``run_pusht_predictor_experiment.run`` which trains all three optimizers from the
same init and writes a fresh, uniquely named sub-folder:

    outputs/<dataTAG>/<YYYYmmdd_HHMMSS_fff>__seed<seed>/
        metrics.csv   run.json

Per fraction it then aggregates the final rows into ``comparison.csv`` /
``comparison_agg.csv`` (mean/std over seeds).

    python src/run_experiment1.py                              # full sweep from the YAML
    python src/run_experiment1.py --only-fraction 0.04         # just 4%
    python src/run_experiment1.py --set train.budget=400 --set seed=[3072]   # quick smoke
    python src/run_experiment1.py --dry-run                    # print the plan
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from src.run_pusht_predictor_experiment import (
    DEFAULT_CONFIG,
    OUTPUT_ROOT,
    build_run_args,
    data_tag,
    load_config,
    run,
)

OUT_ROOT = OUTPUT_ROOT  # outputs/<dataTAG>/<stamp>__seed<seed>/  (no 'experiment1' level)

# Columns pulled from each optimizer's final metric row into comparison.csv.
METRIC_KEYS = (
    "backprop_calls", "status", "early_stopped",
    "val_mse", "train_mse", "train_val_gap",
    "rollout_mse_1", "rollout_mse_3", "rollout_mse_5",
    # collapse diagnostics (val one-step): encoder side + predictor side
    "col_enc_emb_var_mean", "col_enc_emb_std_mean", "col_enc_emb_norm_mean", "col_enc_dead_dim_frac",
    "col_pred_var_mean", "col_target_var_mean", "col_pred_target_var_ratio",
    "col_pred_norm_mean", "col_pred_target_norm_ratio",
    # collapse diagnostics (rollout, max horizon)
    "roll_pred_target_var_ratio", "roll_pred_target_norm_ratio",
    "latent_norm_drift", "target_latent_norm", "pred_latent_variance",
)
# Metrics that get a mean/std over seeds in comparison_agg.csv.
AGG_METRICS = (
    "val_mse", "train_mse", "train_val_gap",
    "rollout_mse_1", "rollout_mse_3", "rollout_mse_5",
    "col_enc_emb_var_mean", "col_enc_dead_dim_frac",
    "col_pred_target_var_ratio", "col_pred_target_norm_ratio",
    "roll_pred_target_var_ratio",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1 driver (single-YAML, no tuning).")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the experiment YAML.")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="Dotlist override, repeatable (e.g. --set train.budget=400).")
    p.add_argument("--only-fraction", type=float, default=None,
                   help="Run just this data fraction instead of every entry in data.fractions.")
    p.add_argument("--device", default=None, help="Override cfg.device.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return p.parse_args()


def _aggregate(fraction: float, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = list(dict.fromkeys(r["optimizer"] for r in rows))
    agg: list[dict[str, Any]] = []
    for name in names:
        ok = [r for r in rows if r["optimizer"] == name and r.get("status") == "ok"]
        entry: dict[str, Any] = {"optimizer": name, "data_fraction": fraction, "n_seeds_ok": len(ok)}
        for metric in AGG_METRICS:
            vals = [float(r[metric]) for r in ok if r.get(metric) is not None]
            entry[f"{metric}_mean"] = statistics.fmean(vals) if vals else None
            entry[f"{metric}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        agg.append(entry)
    return agg


def write_comparison(fraction: float, rows: list[dict[str, Any]], run_dirs: list[Path]) -> None:
    """Write comparison.csv / comparison_agg.csv INTO every run folder from this
    invocation (self-contained; a later run never overwrites an earlier one).
    ``rows``: one dict per (optimizer, seed) final result."""

    fieldnames = ["optimizer", "seed", "data_fraction", *METRIC_KEYS, "error"]
    agg = _aggregate(fraction, rows)
    agg_fields = ["optimizer", "data_fraction", "n_seeds_ok",
                  *[f"{m}_{s}" for m in AGG_METRICS for s in ("mean", "std")]]

    for rd in run_dirs:
        with (rd / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        with (rd / "comparison_agg.csv").open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=agg_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(agg)

    print(f"\n=== {data_tag(fraction)}  (mean over seeds) ===")
    print(f"{'optimizer':<14} {'n_ok':>4} {'val_mse':>11} {'rollout5':>11} {'pred/tgt var':>13} {'enc var':>10}")
    for entry in agg:
        def fmt(key: str) -> str:
            v = entry.get(key)
            return f"{v:.5f}" if isinstance(v, float) else "n/a"
        print(f"{entry['optimizer']:<14} {entry['n_seeds_ok']:>4} {fmt('val_mse_mean'):>11} {fmt('rollout_mse_5_mean'):>11} "
              f"{fmt('col_pred_target_var_ratio_mean'):>13} {fmt('col_enc_emb_var_mean_mean'):>10}")
    for rd in run_dirs:
        print(f"wrote {rd/'comparison.csv'}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.device:
        cfg.device = args.device

    fractions = [args.only_fraction] if args.only_fraction is not None else [float(f) for f in cfg.data.fractions]
    seeds = [int(s) for s in cfg.seed]

    print("=" * 78)
    print(f"Experiment 1  |  fractions={fractions}  seeds={seeds}  optimizers={list(cfg.optimizers)}")
    print(f"  epochs={cfg.train.epochs} budget={cfg.train.budget} batch={cfg.train.batch_size} "
          f"precision={cfg.train.precision} device={cfg.device}")
    print(f"  out: {OUT_ROOT}")
    print("=" * 78)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # No top-level config snapshot (it would be overwritten on every run). Each run
    # folder gets its own verbatim outputs/<dataTAG>/<run>/config.yaml + run.json.
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return

    for fraction in fractions:
        out_dir = OUT_ROOT / data_tag(fraction)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        run_dirs: list[Path] = []
        for seed in seeds:
            print(f"\n---- {data_tag(fraction)}  seed {seed} ----")
            run_args = build_run_args(cfg, data_fraction=fraction, seed=seed)
            run_dir, finals = run(run_args, out_dir)
            run_dirs.append(run_dir)
            for final in finals:
                rows.append({
                    "optimizer": final.get("optimizer"), "seed": seed, "data_fraction": fraction,
                    **{k: final.get(k) for k in METRIC_KEYS}, "error": final.get("error", ""),
                })
        write_comparison(fraction, rows, run_dirs)

    print(f"\n[experiment1] done -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
