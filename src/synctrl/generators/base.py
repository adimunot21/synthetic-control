"""Shared utilities for the three synthetic control-prior generators
(MDP, Bandit, LDS): random-Fourier-feature GP-style function sampling,
feature padding/permutation, and z-scoring.

Every generator exposes:

    get_batch(batch_size, seq_len, num_features, rng) -> (x, y, y_)

following the TabPFN seq-first convention validated in Phase 0
(docs/data-contract-tabpfn.md): x is (seq_len, batch_size, num_features)
float32, y/y_ are (seq_len, batch_size). Each of the `batch_size` tasks in
one call is an independently sampled system (fresh RFF functions / MDP /
policy), not batch_size rollouts of the *same* system -- matching "for each
new task, we draw a generator... and sample one dataset" (paper, Section IV).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RandomFourierFunction:
    """One sampled realization of an RFF-approximated GP draw, mapping
    R^d_in -> R^d_out: f(x) = sum_i w_i * cos(omega_i . x + phi_i), with the
    same (omega, phi) basis shared across output dimensions (so the outputs
    are correlated draws from one latent function, not independent ones --
    the "multi-output" part of the paper's "multi-output Gaussian-process
    prior").
    """

    omega: np.ndarray  # (n_rff, d_in)
    phi: np.ndarray  # (n_rff,)
    weights: np.ndarray  # (n_rff, d_out)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (..., d_in) -> (..., d_out)
        features = np.cos(x @ self.omega.T + self.phi)  # (..., n_rff)
        return features @ self.weights


def sample_rff_function(
    rng: np.random.Generator,
    d_in: int,
    d_out: int,
    n_rff: int,
    lengthscale: float,
) -> RandomFourierFunction:
    omega = rng.normal(scale=1.0 / lengthscale, size=(n_rff, d_in))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_rff)
    weights = rng.normal(scale=1.0 / np.sqrt(n_rff), size=(n_rff, d_out))
    return RandomFourierFunction(omega=omega, phi=phi, weights=weights)


def sample_rff_basis(rng: np.random.Generator, d_in: int, n_rff: int, lengthscale: float) -> tuple[np.ndarray, np.ndarray]:
    """Just the (omega, phi) basis of a random-Fourier-feature map, without
    a GP-sample readout -- used where RFF embeddings themselves are the
    feature representation (e.g. the Bandit generator's bilinear reward),
    not where we need a scalar/vector *function sample* like
    `sample_rff_function` provides.
    """
    omega = rng.normal(scale=1.0 / lengthscale, size=(n_rff, d_in))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_rff)
    return omega, phi


def rff_features(x: np.ndarray, omega: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """x: (..., d_in) -> (..., n_rff)."""
    return np.cos(x @ omega.T + phi)


def pad_and_permute_features(x: np.ndarray, num_features: int, rng: np.random.Generator) -> np.ndarray:
    """x: (seq_len, d). Zero-pad to num_features columns if d < num_features,
    then permute all num_features columns so no input index carries
    consistent meaning across tasks (paper, Section IV-B).
    """
    seq_len, d = x.shape
    assert d <= num_features, f"intrinsic dimensionality {d} exceeds requested num_features {num_features}"
    if d < num_features:
        padding = np.zeros((seq_len, num_features - d), dtype=x.dtype)
        x = np.concatenate([x, padding], axis=1)
    permutation = rng.permutation(num_features)
    return x[:, permutation]


def z_score(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def stack_batch_to_tabpfn_layout(
    x_list: list[np.ndarray], y_list: list[np.ndarray], y_cont_list: list[np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x_list[i]: (seq_len, num_features); y_list[i], y_cont_list[i]: (seq_len,).
    Stacks into TabPFN's (seq_len, batch_size, *) layout.
    """
    x = np.stack(x_list, axis=1)  # (seq_len, batch_size, num_features)
    y = np.stack(y_list, axis=1)  # (seq_len, batch_size)
    y_cont = np.stack(y_cont_list, axis=1)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(y_cont, dtype=torch.float32),
    )
