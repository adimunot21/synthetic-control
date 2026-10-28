import numpy as np
import pytest

from synctrl.sims.cartpole import CartPole
from synctrl.sims.dubins import DubinsCar
from synctrl.sims.pendulum import Pendulum
from synctrl.tasks import rollout_to_task, sample_tasks, task_from_arrays

SYSTEMS = [Pendulum(), CartPole(), DubinsCar()]
SYSTEM_IDS = ["pendulum", "cartpole", "dubins"]


@pytest.mark.parametrize("system", SYSTEMS, ids=SYSTEM_IDS)
@pytest.mark.parametrize("split", ["contiguous", "random"])
def test_task_shapes_and_split_sizes(system, split):
    rng = np.random.default_rng(0)
    rollout = system.rollout(n_steps=100, rng=rng, noise_eps=0.1)
    task = rollout_to_task(system, rollout, split=split, rng=rng, context_frac=0.5)

    assert task.context_x.shape == (50, system.state_dim)
    assert task.query_x.shape == (50, system.state_dim)
    assert task.context_y.shape == (50,)
    assert task.query_y.shape == (50,)
    assert task.action_space_size == system.action_space_size
    assert task.split == split


def test_contiguous_split_preserves_time_order():
    rng = np.random.default_rng(1)
    system = Pendulum()
    rollout = system.rollout(n_steps=20, rng=rng, noise_eps=0.0)
    task = rollout_to_task(system, rollout, split="contiguous", rng=rng, context_frac=0.5)

    # context = first half in time, query = second half, with no overlap and
    # no reordering within each half.
    np.testing.assert_array_equal(task.context_x, rollout.states[:-1][:10])
    np.testing.assert_array_equal(task.query_x, rollout.states[:-1][10:])


def test_random_split_is_a_permutation_of_contiguous_split():
    rng_a = np.random.default_rng(2)
    rng_b = np.random.default_rng(2)
    system = Pendulum()
    rollout = system.rollout(n_steps=40, rng=rng_a, noise_eps=0.1)

    task_contig = rollout_to_task(system, rollout, split="contiguous", rng=rng_a, context_frac=0.5)
    task_random = rollout_to_task(system, rollout, split="random", rng=rng_b, context_frac=0.5)

    # Same underlying (x, y) pairs, just reassigned between context/query.
    all_x_contig = np.concatenate([task_contig.context_x, task_contig.query_x])
    all_x_random = np.concatenate([task_random.context_x, task_random.query_x])
    assert sorted(all_x_contig.flatten().tolist()) == sorted(all_x_random.flatten().tolist())


def test_random_split_actually_shuffles():
    # With 100 steps the probability of the random split landing back on
    # a purely-contiguous assignment by chance is astronomically small.
    rng = np.random.default_rng(3)
    system = Pendulum()
    rollout = system.rollout(n_steps=100, rng=rng, noise_eps=0.1)
    task = rollout_to_task(system, rollout, split="random", rng=rng, context_frac=0.5)

    contiguous_context = rollout.states[:-1][:50]
    assert not np.array_equal(task.context_x, contiguous_context)


def test_task_from_arrays_matches_rollout_to_task():
    """rollout_to_task should now just be task_from_arrays applied to
    rollout.states[:-1]/rollout.actions -- confirm the refactor didn't
    change behavior.
    """
    system = Pendulum()
    rollout = system.rollout(n_steps=40, rng=np.random.default_rng(10), noise_eps=0.1)

    task_a = rollout_to_task(system, rollout, split="contiguous", rng=np.random.default_rng(0), context_frac=0.5)
    task_b = task_from_arrays(
        rollout.states[:-1], rollout.actions, system.action_space_size,
        split="contiguous", rng=np.random.default_rng(0), context_frac=0.5, system_params=rollout.params,
    )

    np.testing.assert_array_equal(task_a.context_x, task_b.context_x)
    np.testing.assert_array_equal(task_a.query_y, task_b.query_y)
    assert task_a.system_params == task_b.system_params


def test_task_from_arrays_casts_float_labels_to_int64():
    """Generator get_batch output (TabPFN convention) is float32 throughout,
    including labels -- task_from_arrays must normalize this, or downstream
    consumers like np.bincount (majority baseline) break (a real bug caught
    while building the RQ2 autocorrelation experiment).
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3)).astype(np.float32)
    y = rng.integers(0, 4, size=20).astype(np.float32)  # float labels, as get_batch produces
    task = task_from_arrays(x, y, action_space_size=4, split="contiguous", rng=rng)
    assert task.context_y.dtype == np.int64
    assert task.query_y.dtype == np.int64


def test_task_from_arrays_without_system_params_defaults_to_empty_dict():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    y = rng.integers(0, 4, size=20)
    task = task_from_arrays(x, y, action_space_size=4, split="contiguous", rng=rng)
    assert task.system_params == {}


@pytest.mark.parametrize("system", SYSTEMS, ids=SYSTEM_IDS)
def test_sample_tasks_uses_fresh_params_per_task(system):
    rng = np.random.default_rng(4)
    tasks = sample_tasks(system, n_tasks=5, n_steps=30, rng=rng, noise_eps=0.1, split="contiguous")
    assert len(tasks) == 5
    param_sets = [tuple(sorted(t.system_params.items())) for t in tasks]
    assert len(set(param_sets)) == 5, "expected independently randomized parameters per task"
