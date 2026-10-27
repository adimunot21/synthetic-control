"""Phase 1 first result: baseline suite vs. the oracle ceiling on our own
held-out known-expert systems, before any pretrained control-prior model
exists. Not yet the RQ1 decision-gate comparison (that needs the trained
model, task #16) -- this establishes what simple baselines can already do,
which the eventual model has to beat to mean anything.
"""

from __future__ import annotations

import numpy as np

from synctrl.baselines import ALL_BASELINES
from synctrl.eval.harness import print_summary_table, run_suite, save_results
from synctrl.sims.cartpole import CartPole
from synctrl.sims.dubins import DubinsCar
from synctrl.sims.pendulum import Pendulum

CONFIG = {
    "n_tasks_per_system": 60,  # enough for reasonably tight bootstrap CIs at this stage
    "n_steps": 200,  # 100 context / 100 query at context_frac=0.5
    "noise_eps": 0.15,  # leaves meaningful headroom below the oracle ceiling
    "splits": ("contiguous", "random"),
    "seed": 0,
}

SYSTEMS = {"pendulum": Pendulum(), "cartpole": CartPole(), "dubins": DubinsCar()}


def main() -> None:
    print("Oracle accuracy ceiling at noise_eps =", CONFIG["noise_eps"])
    for name, system in SYSTEMS.items():
        print(f"  {name}: {system.oracle_accuracy_ceiling(CONFIG['noise_eps']):.4f}")
    print()

    results = run_suite(
        SYSTEMS,
        ALL_BASELINES,
        n_tasks_per_system=CONFIG["n_tasks_per_system"],
        n_steps=CONFIG["n_steps"],
        noise_eps=CONFIG["noise_eps"],
        splits=CONFIG["splits"],
        seed=CONFIG["seed"],
    )
    print_summary_table(results)
    save_results(results, "results/phase1_baseline_suite.json")
    print("\nSaved to results/phase1_baseline_suite.json")


if __name__ == "__main__":
    main()
