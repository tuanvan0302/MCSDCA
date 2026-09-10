"""PushT MPC/CEM planning evaluation for a trained LeWM checkpoint.

Separated from the training script so that training/tuning sweeps never pull in
``stable-worldmodel[env]`` or spin a vectorized environment. Run it on a
``*_full_model.pt`` checkpoint produced by ``run_pusht_predictor_experiment.py``::

    uv run python src/evaluate_planning.py \
        --checkpoint outputs/pusht_predictor_optimizer/<run>/checkpoints/adamw_full_model.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multiprocess_compat import patch_multiprocess_resource_tracker

patch_multiprocess_resource_tracker()

import numpy as np
import torch
from omegaconf import OmegaConf

from src.run_pusht_predictor_experiment import (
    DEFAULT_CONFIG,
    IMAGE_MEAN,
    IMAGE_STD,
    PushTHDF5Sampler,
    initialize_lewm,
    load_config,
)

ROLLOUT_MAX_SEQ_LEN = 8  # history_size (3) + max rollout horizon (5)


class ZScoreProcessor:
    """Minimal sklearn-compatible processor used by WorldModelPolicy."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.detach().cpu().numpy()
        self.std = std.detach().cpu().numpy()

    def transform(self, value: np.ndarray) -> np.ndarray:
        return (value - self.mean) / self.std

    def inverse_transform(self, value: np.ndarray) -> np.ndarray:
        return value * self.std + self.mean


def normalize_planning_image(image: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(image)
    value = value.float() / 255.0 if not value.is_floating_point() else value.float()
    mean = IMAGE_MEAN.view(3, 1, 1).to(value)
    std = IMAGE_STD.view(3, 1, 1).to(value)
    return (value - mean) / std


def select_planning_cases(
    sampler: PushTHDF5Sampler,
    num_episodes: int,
    goal_offset: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    candidates = sampler.val_eps[sampler.ep_len[sampler.val_eps] > goal_offset]
    if len(candidates) < num_episodes:
        raise ValueError(
            f"Planning needs {num_episodes} val episodes with >{goal_offset} steps; only {len(candidates)} exist."
        )
    rng = np.random.default_rng(seed)
    episodes = rng.choice(candidates, size=num_episodes, replace=False)
    starts = [int(rng.integers(0, int(sampler.ep_len[ep]) - goal_offset)) for ep in episodes]
    return [int(ep) for ep in episodes], starts


def evaluate_pusht_planning(
    model: torch.nn.Module,
    data_path: Path,
    sampler: PushTHDF5Sampler,
    episodes: list[int],
    start_steps: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    try:
        import stable_worldmodel as swm
    except ImportError as exc:
        raise RuntimeError("PushT planning requires stable-worldmodel[env].") from exc

    class RecordingCEMSolver(swm.solver.CEMSolver):
        def __init__(self, *solver_args: Any, **solver_kwargs: Any) -> None:
            super().__init__(*solver_args, **solver_kwargs)
            self.final_costs: list[float] = []

        def solve(self, *solve_args: Any, **solve_kwargs: Any) -> dict[str, Any]:
            output = super().solve(*solve_args, **solve_kwargs)
            self.final_costs.extend(float(value) for value in output["costs"])
            return output

    action_block = args.planning_action_block or args.frameskip
    plan_config = swm.PlanConfig(
        horizon=args.planning_horizon,
        receding_horizon=args.planning_receding_horizon,
        history_len=args.history_size,
        action_block=action_block,
    )
    solver = RecordingCEMSolver(
        model=model,
        batch_size=min(args.cem_batch_size, len(episodes)),
        num_samples=args.cem_num_samples,
        var_scale=args.cem_var_scale,
        n_steps=args.cem_iterations,
        topk=args.cem_topk,
        device=device,
        seed=args.seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process={"action": ZScoreProcessor(sampler.action_mean, sampler.action_std)},
        transform={"pixels": normalize_planning_image, "goal": normalize_planning_image},
    )
    dataset = swm.data.HDF5Dataset(path=data_path, keys_to_cache=["action", "proprio", "state"])
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=len(episodes),
        max_episode_steps=2 * args.planning_eval_budget,
        image_shape=(224, 224),
    )
    model.eval()
    world.set_policy(policy)
    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
    ]
    try:
        start = time.perf_counter()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes,
            start_steps=start_steps,
            goal_offset=args.planning_goal_offset,
            eval_budget=args.planning_eval_budget,
            callables=callables,
        )
        eval_time_s = time.perf_counter() - start
    finally:
        world.envs.close()
        if dataset.h5_file is not None:
            dataset.h5_file.close()

    return {
        "cem_horizon": args.planning_horizon,
        "eval_episodes": len(episodes),
        "pusht_score": float(metrics["success_rate"]),
        "cem_cost": float(np.mean(solver.final_costs)) if solver.final_costs else float("nan"),
        "planning_eval_time_s": float(eval_time_s),
        "episode_successes": [bool(value) for value in metrics["episode_successes"]],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    data_path = Path(args.data_path).resolve()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    sampler = PushTHDF5Sampler(
        data_path, args.frameskip, ROLLOUT_MAX_SEQ_LEN, args.action_stats_samples, args.seed, args.train_fraction
    )
    model_cfg = OmegaConf.to_container(load_config(args.config).model, resolve=True)
    model, _ = initialize_lewm(model_cfg, device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    episodes, start_steps = select_planning_cases(
        sampler, args.planning_episodes, args.planning_goal_offset, args.seed + 1
    )
    result = evaluate_pusht_planning(model, data_path, sampler, episodes, start_steps, args, device)

    out_path = checkpoint_path.with_name(checkpoint_path.stem + "_planning.csv")
    row = {k: v for k, v in result.items() if k != "episode_successes"}
    row["checkpoint"] = checkpoint_path.name
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"pusht_score={result['pusht_score']:.4f} cem_cost={result['cem_cost']:.6g} -> {out_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PushT MPC/CEM planning for a trained LeWM checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a *_full_model.pt state dict.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment YAML (for the model block).")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "pusht_expert_train.h5"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--action-stats-samples", type=int, default=20000)
    parser.add_argument("--planning-episodes", type=int, default=50)
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument("--planning-receding-horizon", type=int, default=5)
    parser.add_argument("--planning-action-block", type=int, default=None, help="Defaults to --frameskip.")
    parser.add_argument("--planning-goal-offset", type=int, default=25)
    parser.add_argument("--planning-eval-budget", type=int, default=50)
    parser.add_argument("--cem-num-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--cem-var-scale", type=float, default=1.0)
    parser.add_argument("--cem-batch-size", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.planning_receding_horizon > args.planning_horizon:
        raise ValueError("planning-receding-horizon cannot exceed planning-horizon.")
    if args.planning_horizon < args.history_size:
        raise ValueError("planning-horizon must be at least history-size.")
    if args.cem_topk > args.cem_num_samples or args.cem_topk < 2:
        raise ValueError("cem-topk must be in [2, cem-num-samples].")
    run(args)


if __name__ == "__main__":
    main()
