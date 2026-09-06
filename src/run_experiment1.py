"""One-command runner for Experiment 1 - MCSDCA vs AdamW as a full-model LeWM
PushT predictor optimizer.

For a SINGLE PushT data fraction it runs the whole pipeline end to end:

  1. sanity    - odLD + udLD take a few steps and stay finite
  2. tune odLD - coarse grid over (eta, epsilon/eta ratio, beta0) x >=2 seeds,
                 ranked on the seed mean (divergent beta0*eta corners excluded)
  3. tune udLD - grid over delta, inheriting epsilon/beta0 from the odLD winner
  4. tune AdamW- grid over (lr, weight_decay), auto-ranked
  5. evaluate  - all three optimizers at their winning config, N seeds, one
                 FIXED backprop budget (the same at every data fraction)

The only required argument is ``--data``: the PERCENT of the PushT training set
to use (0.01, 1, 10, 50, 100 ...). Any value < 1 turns on FLOW-TEST MODE - tiny
grids, tiny budgets, one seed - just to prove the pipeline runs.

Examples
--------
    python src/run_experiment1.py --data 0.01
    python src/run_experiment1.py --data 10
    python src/run_experiment1.py --data 50 --reuse-winners outputs/experiment1/data10/winners.json

Defaults are tuned for the rented VM: Windows 10, i7-12700KF, 1x RTX 3090 (24 GB),
56 GB RAM, NVMe SSD.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from src.run_pusht_predictor_experiment import (
    ROLLOUT_HORIZONS,
    TRAINING_PROFILES,
    PushTHDF5Sampler,
    build_parser,
    run,
)
from src.sweep import combos, run_grid, select_best

OPTIMIZERS = ("AdamW", "MCSDCA-odLD", "MCSDCA-udLD")
METRIC_KEYS = (
    "backprop_calls", "val_mse", "train_mse", "train_val_gap",
    "rollout_mse_1", "rollout_mse_3", "rollout_mse_5",
    "latent_norm_drift", "target_latent_norm", "pred_latent_variance", "status",
)
AGG_METRICS = ("val_mse", "train_mse", "train_val_gap", "rollout_mse_1", "rollout_mse_3", "rollout_mse_5")

# Real-run grids (RTX 3090). Flow-test grids are a tiny subset.
#   * eta range shifted up (1e-3 stalls, per 00_MCSDCA_paper_experiments).
#   * epsilon swept as the RATIO epsilon/eta (sweep.make_args sets
#     epsilon = ratio * eta), so the Langevin noise-to-signal regime is the
#     knob, not an eta-dependent absolute.
#   * n_k and gamma_k are NOT swept: they follow the paper's growing schedules
#     n_k = base + floor((k+1)^power), gamma_k = gamma_0 * (k+1)^power.
REAL_GRIDS: dict[str, dict[str, list]] = {
    "odld": {
        "mcsdca_eta": [3.0e-3, 1.0e-2, 3.0e-2],
        "mcsdca_epsilon_ratio": [1.0e-6, 1.0e-2, 1.0e0],
        "mcsdca_beta0": [0.5, 0.9, 0.99],
    },
    "udld": {"mcsdca_delta": [0.03, 0.1, 0.3]},
    "adamw": {"lr": [2.0e-5, 5.0e-5, 1.0e-4], "weight_decay": [0.0, 1.0e-3, 1.0e-2]},
}
# Drop odLD corners whose effective LR (~ beta0 * n_k/2 * eta) diverges from a
# from-scratch init before they waste a rank-budget slot.
REAL_EXCLUDES: dict[str, list[dict]] = {
    "odld": [
        {"mcsdca_eta": 3.0e-2, "mcsdca_beta0": 0.9},
        {"mcsdca_eta": 3.0e-2, "mcsdca_beta0": 0.99},
    ],
    "udld": [],
    "adamw": [],
}
FLOW_GRIDS: dict[str, dict[str, list]] = {
    "odld": {"mcsdca_eta": [1.0e-2], "mcsdca_epsilon_ratio": [1.0e-6, 1.0e0], "mcsdca_beta0": [0.9]},
    "udld": {"mcsdca_delta": [0.1]},
    "adamw": {"lr": [5.0e-5], "weight_decay": [0.0, 1.0e-3]},
}
FLOW_EXCLUDES: dict[str, list[dict]] = {"odld": [], "udld": [], "adamw": []}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def data_tag(percent: float) -> str:
    return f"data{('%g' % percent).replace('.', 'p')}"


def as_float(value: Any) -> float:
    return float(value)


def grid_size(grid: dict[str, list], exclude: list[dict] | None = None) -> int:
    return len(combos(grid, exclude))


def steps_per_epoch(data_path: Path, profile_name: str, frac: float, batch_size: int, seed: int) -> tuple[int, int]:
    """(#train windows, steps/epoch) for the fixed-budget -> epochs-equivalent note."""

    prof = TRAINING_PROFILES[profile_name]
    seq_len = prof.history_size + prof.num_preds
    max_seq = prof.history_size + max(ROLLOUT_HORIZONS)
    sampler = PushTHDF5Sampler(data_path, prof.frameskip, max_seq, 2000, seed, prof.train_fraction)
    ids = sampler.select_window_ids("train", seq_len, frac, seed + 2)
    return len(ids), max(1, math.ceil(len(ids) / batch_size))


def sweep_and_pick(
    kind: str, optimizer: str, base: dict, grid: dict, exclude: list[dict], out_dir: Path
) -> tuple[list[dict], dict]:
    grid_dir = out_dir.parent / "_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / f"{kind}.yaml"
    OmegaConf.save(
        OmegaConf.create({"base": base, "grid": grid, "exclude": exclude, "output": str(out_dir)}),
        grid_path,
    )
    summary = run_grid(grid_path)
    ranked, best = select_best(summary, optimizer)
    if best is None:
        raise SystemExit(
            f"[experiment1] no '{optimizer}' config passed the selection gates in {summary}.\n"
            f"                Widen the grid or raise --tune-budget."
        )
    print(f"[experiment1] {optimizer} winner: {best['run_id']}  "
          f"rollout5={float(best['rollout_mse_5_mean']):.5f} val={float(best['val_mse_mean']):.5f} "
          f"(n_seeds={best['n_seeds']})")
    return ranked, best


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_sanity(common: dict, seed: int, budget: int, out_dir: Path) -> None:
    args = build_parser().parse_args([])
    for key, value in common.items():
        setattr(args, key, value)
    args.optimizers = "MCSDCA-odLD,MCSDCA-udLD"
    args.backprop_budget = budget
    args.eval_interval = max(1, budget // 2)
    args.seed = seed
    args.output_dir = str(out_dir)
    args.run_id = "sanity"
    args.save_checkpoints = False
    bad = [row for row in run(args) if row.get("status") != "ok"]
    if bad:
        raise SystemExit(f"[experiment1] sanity FAILED (optimizer diverged / errored): {bad}")
    print("[experiment1] sanity OK - odLD and udLD both finite\n")


def stage_tune(common: dict, tune_seeds: list[int], tune_budget: int, grids: dict,
               excludes: dict, out_root: Path) -> dict:
    base = {**common, "backprop_budget": tune_budget, "eval_interval": max(1, tune_budget // 4)}

    # odLD is the seed-sensitive sweep (biggest grid); run every seed and let
    # select_best rank on the seed mean. udLD inherits epsilon/beta0 from the
    # odLD winner and AdamW is well characterised, so both use one seed.
    _, odld = sweep_and_pick(
        "odld", "MCSDCA-odLD", {**base, "optimizers": "MCSDCA-odLD"},
        {"seed": list(tune_seeds), **grids["odld"]}, excludes["odld"], out_root / "sweep_odld",
    )
    odld_eta = as_float(odld["mcsdca_eta"])
    shared_beta0 = as_float(odld["mcsdca_beta0"])
    if "mcsdca_epsilon_ratio" in odld:
        shared_eps = as_float(odld["mcsdca_epsilon_ratio"]) * odld_eta
    else:
        shared_eps = as_float(odld["mcsdca_epsilon"])

    _, udld = sweep_and_pick(
        "udld", "MCSDCA-udLD",
        {**base, "optimizers": "MCSDCA-udLD", "seed": tune_seeds[0],
         "mcsdca_epsilon": shared_eps, "mcsdca_beta0": shared_beta0},
        grids["udld"], excludes["udld"], out_root / "sweep_udld",
    )

    _, adamw = sweep_and_pick(
        "adamw", "AdamW", {**base, "optimizers": "AdamW", "seed": tune_seeds[0]},
        grids["adamw"], excludes["adamw"], out_root / "sweep_adamw",
    )

    return {
        "AdamW": {"lr": as_float(adamw["lr"]), "weight_decay": as_float(adamw["weight_decay"])},
        "MCSDCA-odLD": {"mcsdca_eta": odld_eta,
                        "mcsdca_epsilon": shared_eps, "mcsdca_beta0": shared_beta0},
        "MCSDCA-udLD": {"mcsdca_delta": as_float(udld["mcsdca_delta"]),
                        "mcsdca_epsilon": shared_eps, "mcsdca_beta0": shared_beta0},
    }


def stage_evaluate(common: dict, winners: dict, seeds: list[int], eval_budget: int,
                   frac: float, out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        args = build_parser().parse_args([])
        for key, value in common.items():
            setattr(args, key, value)
        args.optimizers = ",".join(OPTIMIZERS)
        args.backprop_budget = eval_budget
        args.eval_interval = max(1, eval_budget // 10)
        args.seed = seed
        args.lr = winners["AdamW"]["lr"]
        args.weight_decay = winners["AdamW"]["weight_decay"]
        args.mcsdca_eta = winners["MCSDCA-odLD"]["mcsdca_eta"]
        args.mcsdca_delta = winners["MCSDCA-udLD"]["mcsdca_delta"]
        args.mcsdca_epsilon = winners["MCSDCA-odLD"]["mcsdca_epsilon"]
        args.mcsdca_beta0 = winners["MCSDCA-odLD"]["mcsdca_beta0"]
        args.output_dir = str(out_dir)
        args.run_id = f"seed{seed}"
        args.save_checkpoints = False
        for final in run(args):
            rows.append({
                "optimizer": final.get("optimizer"), "seed": seed, "data_fraction": frac,
                **{key: final.get(key) for key in METRIC_KEYS},
                "error": final.get("error", ""),
            })
    return rows


# --------------------------------------------------------------------------- #
# Result IO                                                                    #
# --------------------------------------------------------------------------- #
def write_comparison(rows: list[dict], out_dir: Path) -> None:
    fieldnames = ["optimizer", "seed", "data_fraction", *METRIC_KEYS, "error"]
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    agg: list[dict] = []
    for name in OPTIMIZERS:
        ok = [r for r in rows if r["optimizer"] == name and r.get("status") == "ok"]
        entry: dict[str, Any] = {"optimizer": name, "n_seeds_ok": len(ok),
                                 "data_fraction": rows[0]["data_fraction"] if rows else None}
        for metric in AGG_METRICS:
            vals = [float(r[metric]) for r in ok if r.get(metric) is not None]
            entry[f"{metric}_mean"] = statistics.fmean(vals) if vals else None
            entry[f"{metric}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        agg.append(entry)
    agg_fields = ["optimizer", "data_fraction", "n_seeds_ok",
                  *[f"{m}_{s}" for m in AGG_METRICS for s in ("mean", "std")]]
    with (out_dir / "comparison_agg.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg)

    print("\n=== comparison (mean over seeds) ===")
    print(f"{'optimizer':<14} {'seeds_ok':>8} {'val_mse':>12} {'rollout5':>12} {'train_val_gap':>14}")
    for entry in agg:
        v = entry.get("val_mse_mean")
        r5 = entry.get("rollout_mse_5_mean")
        g = entry.get("train_val_gap_mean")
        print(f"{entry['optimizer']:<14} {entry['n_seeds_ok']:>8} "
              f"{('%.6f' % v) if v is not None else 'n/a':>12} "
              f"{('%.6f' % r5) if r5 is not None else 'n/a':>12} "
              f"{('%.6f' % g) if g is not None else 'n/a':>14}")
    print(f"\nwrote {out_dir / 'comparison.csv'}")
    print(f"wrote {out_dir / 'comparison_agg.csv'}")


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def build_parser_e1() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-command Experiment 1 runner (tune + evaluate at one data fraction).")
    p.add_argument("--data", type=float, required=True,
                   help="Percent of the PushT training set (0.01, 1, 10, 50, 100). <1 => flow-test mode.")
    p.add_argument("--seeds", default="3072,3073,3074", help="Comma list of evaluation seeds.")
    p.add_argument("--tune-seed", type=int, default=3072, help="Seed for sanity + udLD/AdamW sweeps.")
    p.add_argument("--tune-seeds", default="3072,3073",
                   help="Comma list of seeds for the odLD coarse sweep (ranked on the seed mean). "
                        "Flow-test mode forces one seed.")
    p.add_argument("--profile", default="small", choices=tuple(TRAINING_PROFILES))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sigreg-num-proj", type=int, default=512)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-threads", type=int, default=0, help="Torch intra-op threads (0 = auto: min(16, ncpu)).")
    p.add_argument("--window-cache", choices=("auto", "gpu", "cpu", "memmap", "off"), default="auto",
                   help="Decoded-window cache: auto picks GPU / CPU-RAM / on-disk memmap by size, else streams.")
    p.add_argument("--cache-max-gb", type=float, default=12.0, help="GPU VRAM budget for the window cache.")
    p.add_argument("--cache-ram-gb", type=float, default=80.0, help="Host RAM budget for the window cache.")
    p.add_argument("--cache-disk-gb", type=float, default=3000.0, help="NVMe budget for the on-disk frame memmap.")
    p.add_argument("--memmap-dir", default=None, help="Dir for decoded-frame memmaps (default <data dir>/.framecache).")
    p.add_argument("--prefetch-depth", type=int, default=3, help="Minibatches read ahead on background threads (0 = off).")
    p.add_argument("--prefetch-workers", type=int, default=2, help="Background reader threads for prefetch.")
    p.add_argument("--tune-budget", type=int, default=None, help="Backprop budget per sweep point (default 3000, flow 60).")
    p.add_argument("--eval-budget", type=int, default=None,
                   help="FIXED backprop budget for the final comparison, same at every fraction (default 8000, flow 120).")
    p.add_argument("--reuse-winners", default=None, help="Path to a winners.json; skips all tuning.")
    p.add_argument("--only", choices=("all", "tune", "eval"), default="all")
    p.add_argument("--skip-sanity", action="store_true")
    p.add_argument("--data-path", default=str(ROOT / "data" / "pusht_expert_train.h5"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "experiment1"))
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return p


def main() -> None:
    args = build_parser_e1().parse_args()
    if not 0.0 < args.data <= 100.0:
        raise SystemExit("--data must be a percent in (0, 100].")

    frac = args.data / 100.0
    flow = args.data < 1.0
    grids = FLOW_GRIDS if flow else REAL_GRIDS
    excludes = FLOW_EXCLUDES if flow else REAL_EXCLUDES
    tune_budget = args.tune_budget or (60 if flow else 3000)
    eval_budget = args.eval_budget or (120 if flow else 8000)
    sanity_budget = 20 if flow else 40
    seeds = [args.tune_seed] if flow else [int(s) for s in args.seeds.split(",") if s.strip()]
    tune_seeds = ([args.tune_seed] if flow
                  else [int(s) for s in args.tune_seeds.split(",") if s.strip()] or [args.tune_seed])
    data_path = Path(args.data_path).resolve()
    out_dir = Path(args.output_dir).resolve() / data_tag(args.data)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "training_profile": args.profile,
        "data_fraction": frac,
        "device": args.device,
        "precision": args.precision,
        "num_threads": args.num_threads,
        "batch_size": args.batch_size,
        "sigreg_num_proj": args.sigreg_num_proj,
        "window_cache": args.window_cache,
        "cache_max_gb": args.cache_max_gb,
        "cache_ram_gb": args.cache_ram_gb,
        "cache_disk_gb": args.cache_disk_gb,
        "memmap_dir": args.memmap_dir,
        "prefetch_depth": args.prefetch_depth,
        "prefetch_workers": args.prefetch_workers,
    }

    try:
        n_windows, spe = steps_per_epoch(data_path, args.profile, frac, args.batch_size, args.tune_seed)
        epochs_equiv = f"{eval_budget / spe:.2f}"
    except Exception as exc:  # noqa: BLE001 - informational only
        n_windows, spe, epochs_equiv = -1, -1, f"? ({exc!r})"

    print("=" * 78)
    print(f"Experiment 1  |  data={args.data}%  (fraction={frac:g})  "
          f"{'FLOW-TEST MODE' if flow else 'real run'}")
    print(f"  profile={args.profile} batch={args.batch_size} precision={args.precision} "
          f"device={args.device} sigreg_num_proj={args.sigreg_num_proj}")
    print(f"  train windows ~= {n_windows:,}  steps/epoch ~= {spe:,}")
    n_odld = grid_size(grids["odld"], excludes["odld"])
    n_udld = grid_size(grids["udld"], excludes["udld"])
    n_adamw = grid_size(grids["adamw"], excludes["adamw"])
    print(f"  tune  : {n_odld} odLD x {len(tune_seeds)} seed(s) {tune_seeds} "
          f"+ {n_udld} udLD + {n_adamw} AdamW @ seed {args.tune_seed}, "
          f"{tune_budget} backprop each")
    print(f"  eval  : {OPTIMIZERS} x seeds {seeds} @ FIXED {eval_budget} backprop "
          f"(~{epochs_equiv} epochs-equiv)")
    print(f"  out   : {out_dir}")
    if args.reuse_winners:
        print(f"  winners: reuse {args.reuse_winners} (tuning skipped)")
    print("=" * 78 + "\n")

    if args.dry_run:
        return

    t0 = time.perf_counter()
    winners_path = out_dir / "winners.json"

    # ---- winners ---------------------------------------------------------- #
    if args.reuse_winners:
        winners = json.loads(Path(args.reuse_winners).read_text(encoding="utf-8"))["winners"]
    elif args.only == "eval":
        if not winners_path.exists():
            raise SystemExit(f"--only eval needs {winners_path} or --reuse-winners.")
        winners = json.loads(winners_path.read_text(encoding="utf-8"))["winners"]
    else:
        if not args.skip_sanity:
            stage_sanity(common, args.tune_seed, sanity_budget, out_dir)
        winners = stage_tune(common, tune_seeds, tune_budget, grids, excludes, out_dir)
        winners_path.write_text(json.dumps({
            "data_percent": args.data, "data_fraction": frac, "flow_test": flow,
            "profile": args.profile, "tune_seed": args.tune_seed, "tune_seeds": tune_seeds,
            "tune_budget": tune_budget, "winners": winners,
        }, indent=2), encoding="utf-8")
        print(f"\n[experiment1] wrote {winners_path}")
        for name, cfg in winners.items():
            print(f"    {name}: {cfg}")
        print()

    if args.only == "tune":
        print(f"[experiment1] done (tune only) in {time.perf_counter() - t0:.0f}s")
        return

    # ---- evaluate ------------------------------------------------------- #
    rows = stage_evaluate(common, winners, seeds, eval_budget, frac, out_dir)
    write_comparison(rows, out_dir)
    print(f"\n[experiment1] done in {time.perf_counter() - t0:.0f}s  ->  {out_dir}")


if __name__ == "__main__":
    main()
