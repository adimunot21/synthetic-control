"""Per-task metrics and paired-bootstrap confidence intervals.

Per docs/00_critique_and_plan.md: "Use paired bootstrap CIs over tasks and
systems... No cherry-picked checkpoints." The unit of resampling is the
*task*, not the individual query point -- query points within one task share
a context and physical parameters, so treating them as independent samples
would understate the true uncertainty. This also directly enables the
"gap" comparisons the plan calls for (contiguous vs. random split, baseline
vs. model) with honest uncertainty, not just point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def task_accuracy(proba: np.ndarray, query_y: np.ndarray) -> float:
    predicted = proba.argmax(axis=1)
    return float(np.mean(predicted == query_y))


def task_nll(proba: np.ndarray, query_y: np.ndarray) -> float:
    n_query = len(query_y)
    true_class_proba = proba[np.arange(n_query), query_y]
    return float(-np.mean(np.log(true_class_proba)))


@dataclass
class BootstrapResult:
    mean: float
    ci_lo: float
    ci_hi: float
    n_tasks: int


def paired_bootstrap_ci(
    per_task_values: list[float] | np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 2000,
    ci: float = 0.95,
) -> BootstrapResult:
    values = np.asarray(per_task_values, dtype=np.float64)
    n_tasks = len(values)
    observed_mean = float(values.mean())

    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        resample_means[i] = values[idx].mean()

    alpha = (1.0 - ci) / 2.0
    ci_lo = float(np.quantile(resample_means, alpha))
    ci_hi = float(np.quantile(resample_means, 1.0 - alpha))
    return BootstrapResult(mean=observed_mean, ci_lo=ci_lo, ci_hi=ci_hi, n_tasks=n_tasks)


def paired_difference_ci(
    values_a: list[float] | np.ndarray,
    values_b: list[float] | np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 2000,
    ci: float = 0.95,
) -> BootstrapResult:
    """CI on the per-task-paired difference (a - b), e.g. model accuracy minus
    a baseline's accuracy on the *same* tasks. Paired because both were
    evaluated on identical context/query splits, which removes task-to-task
    variance that an unpaired comparison would otherwise have to average
    over, giving a tighter, more honest CI on the actual gap.
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    assert values_a.shape == values_b.shape, "paired comparison requires equal-length, task-aligned arrays"
    diffs = values_a - values_b
    return paired_bootstrap_ci(diffs, rng, n_resamples=n_resamples, ci=ci)
