"""Generic tabular-prior baseline generator (paper Section IV-A): reuses
Hollmann et al.'s TabPFN prior rather than our own control-structure
generators, as the paper's own "no embodiment structure" comparison point.

Phase 0 (docs/data-contract-tabpfn.md) found that the full SCM/BNN/GP
`prior_bag` mixture's hyperparameter-resolution path (via
`sample_differentiable` + `get_mlp_prior_hyperparameters`) does not work
out of the box in this tabpfn==0.1.11 install -- some hyperparameters (e.g.
`num_layers`) arrive at `MLP.__init__` as unresolved callables rather than
sampled scalars, and resolving the exact call order `model_builder.py`
relies on is deferred, out of scope for this generator's purpose.

**This generator therefore uses the GP prior (`tabpfn.priors.fast_gp`)
only, not the full three-way SCM/BNN/GP mixture.** It is a genuine
deviation from the paper's generic prior, not a hidden one -- flagged here
and in docs/reimplementation-assumptions.md. It still serves the same role
the paper's generic prior serves: a baseline comparison generator with *no*
control-specific structure (no notion of state-action-reward, no temporal
rollout), just a smooth random function of the input features, discretized
into classes the same way the paper does it (quantile binning, Section
IV-A) so it fits the same (x, y) action-prediction format as the other
three generators.
"""

from __future__ import annotations

import numpy as np
import torch
from tabpfn.priors import fast_gp

from .base import pad_and_permute_features, quantile_bin, stack_batch_to_tabpfn_layout, z_score

CONFIG = {
    "lengthscale_range": (0.1, 3.0),
    "outputscale_range": (0.5, 2.0),
    "noise_range": (0.01, 0.3),
}


def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    rng: np.random.Generator,
    max_actions: int,
    config: dict = CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_list, y_list, y_cont_list = [], [], []
    for _ in range(batch_size):
        hp = {
            "lengthscale": float(rng.uniform(*config["lengthscale_range"])),
            "outputscale": float(rng.uniform(*config["outputscale_range"])),
            "noise": float(rng.uniform(*config["noise_range"])),
        }
        # fast_gp.get_batch uses torch's global RNG internally, not a seeded
        # numpy Generator -- seed it from our rng so the overall pipeline
        # stays reproducible given a fixed top-level seed.
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        x_t, y_cont_t, _ = fast_gp.get_batch(
            batch_size=1, seq_len=seq_len, num_features=num_features, hyperparameters=hp, device="cpu"
        )
        x_task = x_t[:, 0, :].numpy().astype(np.float64)  # (seq_len, num_features)
        y_cont_task = y_cont_t[:, 0].numpy().astype(np.float64)  # (seq_len,)

        y_task = quantile_bin(y_cont_task, n_bins=max_actions)
        # Already exactly num_features columns (no intrinsic-dim gap to pad),
        # so pad_and_permute_features only permutes here -- still applied for
        # a consistent "no input index carries fixed meaning" contract across
        # all four generators feeding the pretraining mixture.
        x_list.append(z_score(pad_and_permute_features(x_task, num_features, rng)))
        y_list.append(y_task.astype(np.float64))
        y_cont_list.append(y_cont_task)

    return stack_batch_to_tabpfn_layout(x_list, y_list, y_cont_list)
