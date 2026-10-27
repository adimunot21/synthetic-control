"""Contextual Bandit generator (paper Section IV-B): single-step,
state-conditioned action selection with temporal structure removed
entirely -- the generator that isolates state-action association on its
own, without the smooth dynamics (LDS) or long-horizon value structure
(MDP) the other two generators contribute.

See docs/reimplementation-assumptions.md for the bilinear-form and noise
details filled in beyond the paper's text.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import pad_and_permute_features, rff_features, sample_rff_basis, stack_batch_to_tabpfn_layout, z_score

CONFIG = {
    "state_dim_range": (4, 100),
    "action_feature_dim_range": (4, 16),
    "min_actions": 4,
    "n_rff": 64,
    "lengthscale_range": (0.5, 3.0),  # per-task sampled smoothness, see base generator assumptions doc
    "bilinear_rank_range": (1, 8),  # "bilinear form of random rank" -- range chosen, not stated by paper
    "temperature_range": (0.1, 2.0),  # random Boltzmann temperature per task
    # Noise added to the *observed* action features so they don't trivially
    # identify which action was chosen (paper's phrase); scale is relative
    # to the action feature vectors' own std within the task.
    "action_feature_noise_relative_scale": 0.3,
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
    n_actions = rng.integers(config["min_actions"], max_actions + 1)

    # Both state and action-feature dims must fit in num_features together
    # (they're concatenated into one input row below). Sample d_state first
    # with its upper bound reduced to always leave room for at least the
    # action-feature dim's minimum, so the combined dim can never overflow
    # num_features regardless of sampling order.
    state_dim_lo, state_dim_hi = config["state_dim_range"]
    action_feat_lo, action_feat_hi = config["action_feature_dim_range"]
    min_required = state_dim_lo + action_feat_lo
    assert num_features >= min_required, (
        f"num_features={num_features} too small for the bandit generator "
        f"(needs at least state_dim_min + action_feat_dim_min = {min_required})"
    )

    d_state_hi = min(state_dim_hi, num_features - action_feat_lo)
    d_state = rng.integers(state_dim_lo, d_state_hi + 1)

    d_action_feat_hi = min(action_feat_hi, num_features - d_state)
    d_action_feat = rng.integers(action_feat_lo, d_action_feat_hi + 1)
    combined_dim = d_state + d_action_feat
    assert combined_dim <= num_features

    lengthscale = rng.uniform(*config["lengthscale_range"])
    n_rff = config["n_rff"]
    omega_s, phi_s = sample_rff_basis(rng, d_state, n_rff, lengthscale)
    omega_a, phi_a = sample_rff_basis(rng, d_action_feat, n_rff, lengthscale)

    rank = rng.integers(*config["bilinear_rank_range"])
    U = rng.normal(size=(n_rff, rank))
    V = rng.normal(size=(n_rff, rank))
    M = U @ V.T  # (n_rff, n_rff), random-rank bilinear form

    action_features = rng.normal(size=(n_actions, d_action_feat))  # fixed per-task action embeddings
    temperature = rng.uniform(*config["temperature_range"])
    noise_std = config["action_feature_noise_relative_scale"] * (action_features.std() + 1e-8)

    states = rng.normal(size=(seq_len, d_state))
    phi_state = rff_features(states, omega_s, phi_s)  # (seq_len, n_rff)
    phi_action = rff_features(action_features, omega_a, phi_a)  # (n_actions, n_rff)

    rewards = phi_state @ M @ phi_action.T  # (seq_len, n_actions)
    probs = _softmax_rows(rewards / temperature)

    actions = np.array([rng.choice(n_actions, p=probs[t]) for t in range(seq_len)])
    chosen_action_features = action_features[actions] + rng.normal(scale=noise_std, size=(seq_len, d_action_feat))

    x = np.concatenate([states, chosen_action_features], axis=1)
    return x, actions.astype(np.float64)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
