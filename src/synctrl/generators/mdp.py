"""Tabular MDP generator (paper Section IV-B): temporal, policy-induced,
goal-directed sequential decision-making, where the state-to-action mapping
reflects long-horizon value rather than any immediately observable signal.

See docs/reimplementation-assumptions.md for the value-iteration horizon,
discount factor, and the "small spatial displacement" reading used here.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import pad_and_permute_features, sample_rff_function, stack_batch_to_tabpfn_layout, z_score

CONFIG = {
    "state_dim_range": (10, 40),
    "n_states_range": (30, 300),
    "min_actions": 4,
    "n_rff": 64,
    "lengthscale_range": (0.3, 1.0),
    "displacement_scale": 0.1,  # fraction of the [0,1]^d unit cube; keeps transitions "small"
    "discount_gamma": 0.95,  # not stated by the paper; see assumptions doc
    "n_value_iterations": 50,  # "truncated backward value iteration" -- truncation horizon chosen
    "temperature_range": (0.1, 2.0),
}


def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    rng: np.random.Generator,
    max_actions: int,
    config: dict = CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_list, y_list = [], []
    for _ in range(batch_size):
        x_task, y_task = _sample_one_task(rng, seq_len, num_features, max_actions, config)
        x_list.append(z_score(pad_and_permute_features(x_task, num_features, rng)))
        y_list.append(y_task)
    return stack_batch_to_tabpfn_layout(x_list, y_list, y_list)


def _sample_one_task(
    rng: np.random.Generator, seq_len: int, num_features: int, max_actions: int, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    d = rng.integers(config["state_dim_range"][0], min(config["state_dim_range"][1], num_features) + 1)
    n_states = rng.integers(*config["n_states_range"])
    n_actions = rng.integers(config["min_actions"], max_actions + 1)
    lengthscale = rng.uniform(*config["lengthscale_range"])

    state_coords = rng.uniform(0.0, 1.0, size=(n_states, d))  # (n_states, d)

    transition = np.empty((n_states, n_actions), dtype=np.int64)
    reward = np.empty((n_states, n_actions), dtype=np.float64)
    for a in range(n_actions):
        f_a = sample_rff_function(rng, d_in=d, d_out=1 + d, n_rff=config["n_rff"], lengthscale=lengthscale)
        out = f_a(state_coords)  # (n_states, 1+d)
        reward[:, a] = out[:, 0]
        next_continuous = state_coords + config["displacement_scale"] * out[:, 1:]
        transition[:, a] = _nearest_state(next_continuous, state_coords)

    V = _truncated_value_iteration(reward, transition, config["discount_gamma"], config["n_value_iterations"])
    Q = reward + config["discount_gamma"] * V[transition]  # (n_states, n_actions)
    temperature = rng.uniform(*config["temperature_range"])
    policy_probs = _softmax_rows(Q / temperature)

    states_x = np.empty((seq_len, d), dtype=np.float64)
    actions = np.empty(seq_len, dtype=np.int64)
    s = rng.integers(0, n_states)
    for t in range(seq_len):
        a = rng.choice(n_actions, p=policy_probs[s])
        states_x[t] = state_coords[s]
        actions[t] = a
        s = transition[s, a]

    return states_x, actions.astype(np.float64)


def _nearest_state(queries: np.ndarray, state_coords: np.ndarray) -> np.ndarray:
    # queries: (n_states, d), state_coords: (n_states, d) -> (n_states,) nearest index.
    # n_states <= 300 here, so a dense distance matrix is cheap; no KD-tree needed.
    dists = np.linalg.norm(queries[:, None, :] - state_coords[None, :, :], axis=2)
    return dists.argmin(axis=1)


def _truncated_value_iteration(reward: np.ndarray, transition: np.ndarray, gamma: float, n_iters: int) -> np.ndarray:
    n_states, n_actions = reward.shape
    V = np.zeros(n_states)
    for _ in range(n_iters):
        Q = reward + gamma * V[transition]
        V = Q.max(axis=1)
    return V


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
