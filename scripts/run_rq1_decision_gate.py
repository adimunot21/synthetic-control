"""Phase 1 RQ1 decision gate: does the pretrained control-prior model beat
the baseline suite on our held-out known-expert systems, given identical
context? (docs/00_critique_and_plan.md Phase 1.)

Loads the latest smoke-scale checkpoint, evaluates it through the exact
same harness/tasks/metrics as the baseline-only run
(results/phase1_baseline_suite.json), and reports the paired gap plainly --
per the project's explicit instruction to say so if the effect doesn't
survive against simple baselines, not push on regardless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from synctrl.baselines import ALL_BASELINES
from synctrl.eval.harness import print_summary_table, run_suite, save_results
from synctrl.eval.metrics import paired_difference_ci
from synctrl.model import ControlPriorModel, build_model
from synctrl.sims.cartpole import CartPole
from synctrl.sims.dubins import DubinsCar
from synctrl.sims.pendulum import Pendulum

# Must match scripts/run_baseline_suite_phase1.py exactly so both runs score
# the *same* tasks -- that's what makes the paired comparison meaningful.
EVAL_CONFIG = {
    "n_tasks_per_system": 60,
    "n_steps": 200,
    "noise_eps": 0.15,
    "splits": ("contiguous", "random"),
    "seed": 0,
}

SYSTEMS = {"pendulum": Pendulum(), "cartpole": CartPole(), "dubins": DubinsCar()}


def load_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> ControlPriorModel:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}; run scripts/train_control_prior_smoke.py first")
    latest = checkpoints[-1]
    print(f"Loading checkpoint: {latest}")

    payload = torch.load(latest, map_location="cpu", weights_only=False)
    config = payload["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        num_features=config["num_features"],
        n_classes=config["max_actions"],
        ninp=config["ninp"],
        nhead=config["nhead"],
        nhid=config["nhid"],
        nlayers=config["nlayers"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    print(f"Checkpoint is from step {payload['step']}, trained with config: {config}")

    return ControlPriorModel(
        model, num_features=config["num_features"], n_out=config["max_actions"], device=device,
        rng=np.random.default_rng(0),
    )


def main() -> None:
    model_predictor = load_latest_checkpoint()
    predictors = ALL_BASELINES + [model_predictor]

    results = run_suite(
        SYSTEMS, predictors,
        n_tasks_per_system=EVAL_CONFIG["n_tasks_per_system"],
        n_steps=EVAL_CONFIG["n_steps"],
        noise_eps=EVAL_CONFIG["noise_eps"],
        splits=EVAL_CONFIG["splits"],
        seed=EVAL_CONFIG["seed"],
    )
    print_summary_table(results)
    save_results(results, "results/phase1_rq1_decision_gate.json")

    print("\n--- RQ1 decision gate: model vs. best baseline, paired per-task accuracy gap ---")
    rng = np.random.default_rng(0)
    any_baseline_wins = False
    for system_name, by_split in results.items():
        for split, by_predictor in by_split.items():
            model_accs = [r.accuracy for r in by_predictor["control_prior_model"]["per_task"]]
            best_baseline_name, best_baseline_accs = max(
                ((b.name, [r.accuracy for r in by_predictor[b.name]["per_task"]]) for b in ALL_BASELINES),
                key=lambda item: np.mean(item[1]),
            )
            gap = paired_difference_ci(model_accs, best_baseline_accs, rng)
            verdict = "model ahead" if gap.ci_lo > 0 else ("baseline ahead" if gap.ci_hi < 0 else "no significant gap")
            if gap.ci_hi <= 0:
                any_baseline_wins = True
            print(
                f"{system_name:<10} {split:<11} model - best_baseline({best_baseline_name}): "
                f"{gap.mean:+.3f} [{gap.ci_lo:+.3f}, {gap.ci_hi:+.3f}]  -> {verdict}"
            )

    print()
    if any_baseline_wins:
        print(
            "DECISION GATE: at least one system/split shows a baseline matching or beating "
            "the smoke-scale control-prior model. Per the project plan, reporting this plainly "
            "rather than proceeding to scale up -- see the summary above for exactly which cases."
        )
    else:
        print("DECISION GATE: the model beats every baseline on every system/split tested here.")


if __name__ == "__main__":
    main()
