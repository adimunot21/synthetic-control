import numpy as np
import pytest

from synctrl.sims.cartpole import CartPole
from synctrl.sims.dubins import DubinsCar
from synctrl.sims.pendulum import Pendulum
from synctrl.tasks import rollout_to_task, sample_tasks

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


@pytest.mark.parametrize("system", SYSTEMS, ids=SYSTEM_IDS)
def test_sample_tasks_uses_fresh_params_per_task(system):
    rng = np.random.default_rng(4)
    tasks = sample_tasks(system, n_tasks=5, n_steps=30, rng=rng, noise_eps=0.1, split="contiguous")
    assert len(tasks) == 5
    param_sets = [tuple(sorted(t.system_params.items())) for t in tasks]
    assert len(set(param_sets)) == 5, "expected independently randomized parameters per task"
