import numpy as np
import torch

from synctrl.generators import tabular_prior

MAX_ACTIONS = 10


def test_get_batch_shapes_and_dtypes():
    rng = np.random.default_rng(0)
    batch_size, seq_len, num_features = 3, 40, 12
    x, y, y_ = tabular_prior.get_batch(batch_size, seq_len, num_features, rng, max_actions=MAX_ACTIONS)

    assert x.shape == (seq_len, batch_size, num_features)
    assert x.dtype == torch.float32
    assert y.shape == (seq_len, batch_size)
    assert torch.all(y >= 0) and torch.all(y < MAX_ACTIONS)


def test_quantile_bin_produces_roughly_balanced_classes():
    rng = np.random.default_rng(1)
    y_continuous = rng.normal(size=2000)
    labels = tabular_prior.quantile_bin(y_continuous, n_bins=10)

    assert labels.min() == 0
    assert labels.max() == 9
    counts = np.bincount(labels, minlength=10)
    # Quantile binning should split into near-equal-sized groups by construction.
    assert counts.min() > 150  # 2000/10 = 200 expected per bin


def test_y_and_y_continuous_are_consistently_ordered():
    """The discrete label should be monotonic in the continuous value it
    came from (quantile binning preserves rank order), confirming y_ (the
    continuous target) and y (its discretization) stay consistent -- this
    matters for the RCD metric downstream.
    """
    rng = np.random.default_rng(2)
    x, y, y_cont = tabular_prior.get_batch(batch_size=1, seq_len=100, num_features=10, rng=rng, max_actions=8)
    y = y[:, 0].numpy()
    y_cont = y_cont[:, 0].numpy()

    order = np.argsort(y_cont)
    assert np.all(np.diff(y[order]) >= 0), "discrete labels should be non-decreasing in sorted continuous order"
