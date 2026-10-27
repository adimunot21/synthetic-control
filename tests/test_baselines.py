import numpy as np
import pytest

from synctrl.baselines import ALL_BASELINES
from synctrl.sims.pendulum import Pendulum
from synctrl.tasks import sample_tasks


@pytest.mark.parametrize("baseline", ALL_BASELINES, ids=[b.name for b in ALL_BASELINES])
def test_predict_proba_is_a_valid_distribution(baseline):
    rng = np.random.default_rng(0)
    system = Pendulum()
    task = sample_tasks(system, n_tasks=1, n_steps=60, rng=rng, noise_eps=0.1, split="contiguous")[0]

    proba = baseline.predict_proba(task.context_x, task.context_y, task.query_x, task.action_space_size)

    assert proba.shape == (task.query_x.shape[0], task.action_space_size)
    assert np.all(proba > 0), "smoothing should keep every probability strictly positive (finite NLL)"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("baseline", ALL_BASELINES, ids=[b.name for b in ALL_BASELINES])
def test_handles_single_class_context_without_erroring(baseline):
    # Degenerate but realistic: a short or unlucky rollout where the expert
    # (or noise) never switches actions within the context window.
    context_x = np.random.default_rng(1).normal(size=(10, 3))
    context_y = np.zeros(10, dtype=np.int64)
    query_x = np.random.default_rng(2).normal(size=(5, 3))
    n_classes = 4

    proba = baseline.predict_proba(context_x, context_y, query_x, n_classes)

    assert proba.shape == (5, n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # The only observed class should get (by far) the most mass.
    assert np.all(proba[:, 0] > proba[:, 1:].max(axis=1))


def test_majority_baseline_tracks_context_frequency():
    from synctrl.baselines.majority import MajorityClassBaseline

    context_x = np.zeros((10, 2))
    context_y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 2])  # class 0 dominates
    query_x = np.zeros((3, 2))

    proba = MajorityClassBaseline().predict_proba(context_x, context_y, query_x, n_classes=3)
    predicted = proba.argmax(axis=1)
    assert np.all(predicted == 0)


def test_knn_predicts_locally_not_globally():
    from synctrl.baselines.knn import KNNBaseline

    # Two well-separated clusters, different labels: a method using state
    # should get this right; majority-class would not.
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-5.0, scale=0.1, size=(20, 2))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(20, 2))
    context_x = np.concatenate([cluster_a, cluster_b])
    context_y = np.concatenate([np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64)])

    query_x = np.array([[-5.0, 0.0], [5.0, 0.0]])
    proba = KNNBaseline(k=5).predict_proba(context_x, context_y, query_x, n_classes=2)

    assert proba[0].argmax() == 0
    assert proba[1].argmax() == 1
