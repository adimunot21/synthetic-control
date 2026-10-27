"""Ties simulators -> tasks -> predictors (baselines or a pretrained model)
-> metrics together, and runs the Phase 1 RQ1 comparison across systems and
split protocols.

Results are plain nested dicts/dataclasses, saved as JSON -- per the
project's logging decision (plain files, no external dashboard service).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from ..sims.base import KnownExpertSystem
from ..tasks import Task, sample_tasks
from .metrics import BootstrapResult, paired_bootstrap_ci, paired_difference_ci, task_accuracy, task_nll

PredictFn = Callable[[Task], np.ndarray]  # Task -> (n_query, n_classes) probabilities


class Predictor(Protocol):
    name: str

    def predict_proba(
        self, context_x: np.ndarray, context_y: np.ndarray, query_x: np.ndarray, n_classes: int
    ) -> np.ndarray: ...


def as_predict_fn(predictor: Predictor) -> PredictFn:
    def fn(task: Task) -> np.ndarray:
        return predictor.predict_proba(task.context_x, task.context_y, task.query_x, task.action_space_size)

    return fn


@dataclass
class TaskMetricRecord:
    accuracy: float
    nll: float


def evaluate_on_tasks(predict_fn: PredictFn, tasks: list[Task]) -> list[TaskMetricRecord]:
    records = []
    for task in tasks:
        proba = predict_fn(task)
        records.append(TaskMetricRecord(accuracy=task_accuracy(proba, task.query_y), nll=task_nll(proba, task.query_y)))
    return records


@dataclass
class AggregatedResult:
    accuracy: BootstrapResult
    nll: BootstrapResult
    n_tasks: int


def aggregate(records: list[TaskMetricRecord], rng: np.random.Generator) -> AggregatedResult:
    accs = [r.accuracy for r in records]
    nlls = [r.nll for r in records]
    return AggregatedResult(
        accuracy=paired_bootstrap_ci(accs, rng),
        nll=paired_bootstrap_ci(nlls, rng),
        n_tasks=len(records),
    )


def run_suite(
    systems: dict[str, KnownExpertSystem],
    predictors: list[Predictor],
    n_tasks_per_system: int,
    n_steps: int,
    noise_eps: float,
    splits: tuple[str, ...],
    seed: int,
) -> dict:
    """Returns results[system_name][split][predictor_name] = {
        "aggregate": AggregatedResult, "per_task": list[TaskMetricRecord]
    }. Keeping per-task records around (not just the aggregate) is what lets
    us later compute paired model-vs-baseline gaps on the *same* tasks.
    """
    results: dict = {}
    for system_name, system in systems.items():
        results[system_name] = {}
        for split in splits:
            task_rng = np.random.default_rng(seed)
            tasks = sample_tasks(
                system, n_tasks=n_tasks_per_system, n_steps=n_steps, rng=task_rng, noise_eps=noise_eps, split=split
            )
            results[system_name][split] = {}
            for predictor in predictors:
                agg_rng = np.random.default_rng(seed)
                records = evaluate_on_tasks(as_predict_fn(predictor), tasks)
                results[system_name][split][predictor.name] = {
                    "aggregate": aggregate(records, agg_rng),
                    "per_task": records,
                }
    return results


def results_to_jsonable(results: dict) -> dict:
    out: dict = {}
    for system_name, by_split in results.items():
        out[system_name] = {}
        for split, by_predictor in by_split.items():
            out[system_name][split] = {}
            for predictor_name, payload in by_predictor.items():
                out[system_name][split][predictor_name] = {
                    "aggregate": {
                        "accuracy": asdict(payload["aggregate"].accuracy),
                        "nll": asdict(payload["aggregate"].nll),
                        "n_tasks": payload["aggregate"].n_tasks,
                    },
                    "per_task": [asdict(r) for r in payload["per_task"]],
                }
    return out


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_to_jsonable(results), f, indent=2)


def print_summary_table(results: dict) -> None:
    header = f"{'system':<10} {'split':<11} {'predictor':<20} {'accuracy':>22} {'nll':>22}"
    print(header)
    print("-" * len(header))
    for system_name, by_split in results.items():
        for split, by_predictor in by_split.items():
            for predictor_name, payload in by_predictor.items():
                acc = payload["aggregate"].accuracy
                nll = payload["aggregate"].nll
                acc_str = f"{acc.mean:.3f} [{acc.ci_lo:.3f}, {acc.ci_hi:.3f}]"
                nll_str = f"{nll.mean:.3f} [{nll.ci_lo:.3f}, {nll.ci_hi:.3f}]"
                print(f"{system_name:<10} {split:<11} {predictor_name:<20} {acc_str:>22} {nll_str:>22}")
