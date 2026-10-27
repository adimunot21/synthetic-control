"""Unit tests for the Phase 0 known-expert simulators.

Checks, per system: (1) the deterministic expert policy actually accomplishes
its control objective (swing-up / balance / reach-goal) from the system's
sampled initial-state distribution, and (2) empirical action-match rate under
epsilon-greedy noise converges to the closed-form oracle ceiling in
KnownExpertSystem.oracle_accuracy_ceiling.
"""

import numpy as np
import pytest

from synctrl.sims.cartpole import CartPole
from synctrl.sims.dubins import DubinsCar
from synctrl.sims.pendulum import Pendulum, _wrap_angle

SYSTEMS = [Pendulum(), CartPole(), DubinsCar()]
SYSTEM_IDS = ["pendulum", "cartpole", "dubins"]


@pytest.mark.parametrize("system", SYSTEMS, ids=SYSTEM_IDS)
def test_rollout_shapes(system):
    rng = np.random.default_rng(42)
    n_steps = 50
    rollout = system.rollout(n_steps=n_steps, rng=rng, noise_eps=0.1)

    assert rollout.states.shape == (n_steps + 1, system.state_dim)
    assert rollout.actions.shape == (n_steps,)
    assert rollout.actions.min() >= 0
    assert rollout.actions.max() < system.action_space_size
    assert np.all(np.isfinite(rollout.states))


@pytest.mark.parametrize(
    "system,noise_eps",
    [(Pendulum(), 0.0), (Pendulum(), 0.2), (Pendulum(), 0.5)],
    ids=["eps=0.0", "eps=0.2", "eps=0.5"],
)
def test_pendulum_oracle_ceiling(system, noise_eps):
    _assert_oracle_ceiling_matches(system, noise_eps, n_trials=30000, seed=0)


@pytest.mark.parametrize("noise_eps", [0.0, 0.15, 0.4], ids=["eps=0.0", "eps=0.15", "eps=0.4"])
def test_cartpole_oracle_ceiling(noise_eps):
    _assert_oracle_ceiling_matches(CartPole(), noise_eps, n_trials=30000, seed=1)


@pytest.mark.parametrize("noise_eps", [0.0, 0.25, 0.5], ids=["eps=0.0", "eps=0.25", "eps=0.5"])
def test_dubins_oracle_ceiling(noise_eps):
    _assert_oracle_ceiling_matches(DubinsCar(), noise_eps, n_trials=30000, seed=2)


def _assert_oracle_ceiling_matches(system, noise_eps, n_trials, seed):
    rng = np.random.default_rng(seed)
    matches = 0
    for _ in range(n_trials):
        params = system.sample_params(rng)
        state = system.sample_initial_state(rng, params)
        expert_a = system.expert_action(state, params)
        if rng.random() < noise_eps:
            observed_a = int(rng.integers(0, system.action_space_size))
        else:
            observed_a = expert_a
        matches += int(observed_a == expert_a)

    empirical = matches / n_trials
    analytic = system.oracle_accuracy_ceiling(noise_eps)
    # 3-sigma binomial tolerance at n_trials=30000 is well under 1%; 2% leaves
    # headroom while still catching a wrong formula or an off-by-one bug.
    assert abs(empirical - analytic) < 0.02, (
        f"empirical={empirical:.4f} vs analytic={analytic:.4f} for noise_eps={noise_eps}"
    )


def test_pendulum_swingup_reaches_upright():
    rng = np.random.default_rng(7)
    system = Pendulum()
    for _ in range(10):
        rollout = system.rollout(n_steps=600, rng=rng, noise_eps=0.0)
        final_angle_from_upright = _wrap_angle(rollout.states[-1, 0] - np.pi)
        assert abs(final_angle_from_upright) < 0.3


def test_cartpole_balance_keeps_pole_up():
    rng = np.random.default_rng(8)
    system = CartPole()
    for _ in range(10):
        rollout = system.rollout(n_steps=500, rng=rng, noise_eps=0.0)
        assert np.max(np.abs(rollout.states[:, 2])) < 0.5


def test_dubins_reaches_goal():
    # A Dubins car can't stop, so the controller converges to orbiting the
    # goal rather than resting on it; check the minimum distance over the
    # whole trajectory, not the final-step distance (see dubins.py docstring).
    rng = np.random.default_rng(9)
    system = DubinsCar()
    for _ in range(10):
        rollout = system.rollout(n_steps=500, rng=rng, noise_eps=0.0)
        params = rollout.params
        goal_xy = np.array([params["goal_x"], params["goal_y"]])
        dists_to_goal = np.linalg.norm(rollout.states[:, :2] - goal_xy, axis=1)
        assert dists_to_goal.min() < 2.0


@pytest.mark.parametrize("system", SYSTEMS, ids=SYSTEM_IDS)
def test_sample_params_stays_in_documented_ranges(system):
    rng = np.random.default_rng(123)
    for _ in range(200):
        params = system.sample_params(rng)
        for key, value in params.items():
            range_key = f"{key}_range"
            if range_key in system.config:
                lo, hi = system.config[range_key]
                assert lo <= value <= hi, f"{key}={value} outside documented range {system.config[range_key]}"
