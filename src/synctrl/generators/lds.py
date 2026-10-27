"""Linear Dynamical System generator (paper Section IV-B): continuous
dynamics with high temporal autocorrelation, x_{t+1} = A*x_t + b_a + noise,
rolled out under a random linear-softmax policy. This is the generator the
paper's own ablation (Section V-C) found strongest and attributes to local
temporal continuity rather than long-horizon value structure -- exactly the
mechanism docs/00_critique_and_plan.md's skeptical hypothesis is about, so
getting the autocorrelation property right and testable matters more here
than for the other two generators.

See docs/reimplementation-assumptions.md for every filled-in detail (decay
rate range, policy class, noise scale).
"""

from __future__ import annotations

import numpy as np
import torch

from .base import pad_and_permute_features, stack_batch_to_tabpfn_layout, z_score

CONFIG = {
    "state_dim_range": (4, 100),  # per paper, LDS state dimension range
    "min_actions": 4,
    # Decay rates "near one" (paper's phrase, unquantified) -- high enough
    # for strong autocorrelation, below 1 for stability. See
    # docs/reimplementation-assumptions.md.
    "decay_rate_range": (0.90, 0.999),
    "action_displacement_scale": 1.0,  # std of each action's fixed displacement b_a
    "process_noise_std_range": (0.01, 0.1),  # "small Gaussian noise" (paper), range chosen
    "policy_logit_scale": 1.0,  # std of linear-softmax policy weights W, b
    "init_state_std": 1.0,
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
    # LDS actions are discrete-native (no continuous action backing to track
    # separately for RCD) -- y_ duplicates y, same convention fast_gp uses
    # when there's no discretization step (see docs/data-contract-tabpfn.md).
    return stack_batch_to_tabpfn_layout(x_list, y_list, y_list)


def _sample_one_task(
    rng: np.random.Generator, seq_len: int, num_features: int, max_actions: int, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    d = rng.integers(config["state_dim_range"][0], min(config["state_dim_range"][1], num_features) + 1)
    n_actions = rng.integers(config["min_actions"], max_actions + 1)

    decay = rng.uniform(*config["decay_rate_range"], size=d)
    A = np.diag(decay)
    b_actions = rng.normal(scale=config["action_displacement_scale"], size=(n_actions, d))
    noise_std = rng.uniform(*config["process_noise_std_range"])

    policy_W = rng.normal(scale=config["policy_logit_scale"], size=(n_actions, d))
    policy_b = rng.normal(scale=config["policy_logit_scale"], size=n_actions)

    states = np.empty((seq_len, d), dtype=np.float64)
    actions = np.empty(seq_len, dtype=np.int64)
    state = rng.normal(scale=config["init_state_std"], size=d)

    for t in range(seq_len):
        logits = policy_W @ state + policy_b
        probs = _softmax(logits)
        action = rng.choice(n_actions, p=probs)

        states[t] = state
        actions[t] = action
        state = A @ state + b_actions[action] + rng.normal(scale=noise_std, size=d)

    return states, actions.astype(np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
