import json

import numpy as np

from synctrl.baselines.majority import MajorityClassBaseline
from synctrl.eval.harness import run_suite, save_results
from synctrl.eval.metrics import paired_bootstrap_ci, paired_difference_ci, task_accuracy, task_nll
from synctrl.sims.pendulum import Pendulum


def test_task_accuracy_basic():
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    query_y = np.array([0, 1, 0])  # third row: predicted class 1, true class 0 -> wrong
    assert task_accuracy(proba, query_y) == 2 / 3


def test_task_nll_basic():
    proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    query_y = np.array([0, 1])
    assert abs(task_nll(proba, query_y) - (-np.log(0.5))) < 1e-9


def test_task_nll_matches_perfect_prediction():
    proba = np.array([[0.999, 0.001], [0.001, 0.999]])
    query_y = np.array([0, 1])
    assert task_nll(proba, query_y) < 0.01


def test_bootstrap_ci_brackets_true_mean_for_known_distribution():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.7, scale=0.05, size=500)
    result = paired_bootstrap_ci(values, rng)
    assert result.ci_lo < result.mean < result.ci_hi
    assert abs(result.mean - 0.7) < 0.02


def test_bootstrap_ci_narrows_with_more_tasks():
    rng = np.random.default_rng(0)
    small = paired_bootstrap_ci(rng.normal(0.5, 0.1, size=10), np.random.default_rng(1))
    large = paired_bootstrap_ci(rng.normal(0.5, 0.1, size=1000), np.random.default_rng(1))
    assert (large.ci_hi - large.ci_lo) < (small.ci_hi - small.ci_lo)


def test_paired_difference_ci_zero_for_identical_arrays():
    rng = np.random.default_rng(0)
    values = rng.normal(0.6, 0.1, size=100)
    result = paired_difference_ci(values, values, rng)
    assert result.mean == 0.0
    assert result.ci_lo <= 0.0 <= result.ci_hi


def test_run_suite_end_to_end_with_majority_baseline(tmp_path):
    systems = {"pendulum": Pendulum()}
    predictors = [MajorityClassBaseline()]
    results = run_suite(
        systems, predictors, n_tasks_per_system=8, n_steps=40, noise_eps=0.1,
        splits=("contiguous", "random"), seed=0,
    )

    assert set(results.keys()) == {"pendulum"}
    assert set(results["pendulum"].keys()) == {"contiguous", "random"}
    agg = results["pendulum"]["contiguous"]["majority"]["aggregate"]
    assert 0.0 <= agg.accuracy.mean <= 1.0
    assert agg.nll.mean > 0.0
    assert agg.accuracy.n_tasks == 8

    out_path = tmp_path / "results.json"
    save_results(results, out_path)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["pendulum"]["contiguous"]["majority"]["aggregate"]["n_tasks"] == 8
