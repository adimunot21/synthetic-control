"""Shared interface for known-expert simulated systems.

Each system exposes a deterministic physics step and a deterministic expert
policy. Rollouts inject epsilon-greedy noise on top of the expert policy: with
probability `noise_eps` a uniformly random action is taken instead of the
expert's choice. This gives a closed-form oracle accuracy ceiling,

    P(observed action == expert action) = (1 - noise_eps) + noise_eps / n_actions

which `tests/test_sims.py` checks empirically against rollouts. No predictor
that only sees (state, action) pairs can exceed this ceiling in expectation,
since the noise is independent of state.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class Rollout:
    states: np.ndarray  # (T + 1, state_dim), states[t] is the state before action[t]
    actions: np.ndarray  # (T,) int, discrete action indices
    params: dict  # the randomized physical parameters used for this rollout


class KnownExpertSystem(abc.ABC):
    """A discrete-time dynamical system with a fixed, known expert policy."""

    action_space_size: int
    state_dim: int

    @abc.abstractmethod
    def sample_params(self, rng: np.random.Generator) -> dict:
        """Randomize physical parameters within the system's documented ranges."""

    @abc.abstractmethod
    def sample_initial_state(self, rng: np.random.Generator, params: dict) -> np.ndarray:
        """Draw a random initial state."""

    @abc.abstractmethod
    def step(self, state: np.ndarray, action: int, params: dict) -> np.ndarray:
        """Deterministic one-step dynamics update."""

    @abc.abstractmethod
    def expert_action(self, state: np.ndarray, params: dict) -> int:
        """The fixed expert policy's choice of discrete action at `state`."""

    def oracle_accuracy_ceiling(self, noise_eps: float) -> float:
        return (1.0 - noise_eps) + noise_eps / self.action_space_size

    def rollout(
        self,
        n_steps: int,
        rng: np.random.Generator,
        noise_eps: float = 0.0,
        params: dict | None = None,
    ) -> Rollout:
        if params is None:
            params = self.sample_params(rng)
        state = self.sample_initial_state(rng, params)

        states = np.empty((n_steps + 1, self.state_dim), dtype=np.float64)
        actions = np.empty(n_steps, dtype=np.int64)
        states[0] = state

        for t in range(n_steps):
            expert_a = self.expert_action(state, params)
            if rng.random() < noise_eps:
                action = int(rng.integers(0, self.action_space_size))
            else:
                action = expert_a
            state = self.step(state, action, params)
            states[t + 1] = state
            actions[t] = action

        return Rollout(states=states, actions=actions, params=params)
