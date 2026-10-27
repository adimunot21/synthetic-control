"""Statistical-property tests for the three reimplemented generators, per
the global testing rule ("unit tests for... generators (statistical
properties)"). These don't check exact reproduction of the paper (not
possible without its code/checkpoints) -- they check that each generator
actually has the structural property Section IV-B attributes to it.
"""

import numpy as np
import pytest
import torch

from synctrl.generators import bandit, lds, mdp

MAX_ACTIONS = 10


@pytest.mark.parametrize(
    "module", [mdp, bandit, lds], ids=["mdp", "bandit", "lds"]
)
def test_get_batch_shapes_and_dtypes(module):
    rng = np.random.default_rng(0)
    batch_size, seq_len, num_features = 3, 40, 20
    x, y, y_ = module.get_batch(batch_size, seq_len, num_features, rng, max_actions=MAX_ACTIONS)

    assert isinstance(x, torch.Tensor) and x.shape == (seq_len, batch_size, num_features)
    assert x.dtype == torch.float32
    assert y.shape == (seq_len, batch_size)
    assert y_.shape == (seq_len, batch_size)
    assert torch.all(y >= 0) and torch.all(y < MAX_ACTIONS)


@pytest.mark.parametrize("module", [mdp, bandit, lds], ids=["mdp", "bandit", "lds"])
def test_get_batch_is_zscored_per_task(module):
    rng = np.random.default_rng(1)
    x, _, _ = module.get_batch(batch_size=2, seq_len=200, num_features=15, rng=rng, max_actions=MAX_ACTIONS)
    for b in range(x.shape[1]):
        col_means = x[:, b, :].mean(dim=0)
        col_stds = x[:, b, :].std(dim=0)
        # Zero-padded columns have std 0 and are skipped (z_score leaves them
        # at 0, not NaN -- see base.z_score's eps guard); only check columns
        # with nonzero variance.
        active = col_stds > 1e-4
        assert torch.allclose(col_means[active], torch.zeros_like(col_means[active]), atol=0.3)
        assert torch.allclose(col_stds[active], torch.ones_like(col_stds[active]), atol=0.3)


@pytest.mark.parametrize("module", [mdp, bandit, lds], ids=["mdp", "bandit", "lds"])
def test_feature_padding_when_num_features_exceeds_intrinsic_dim(module):
    # A large num_features forces zero-padding for generators whose sampled
    # intrinsic dimension is smaller; shouldn't error, and padded (post-
    # permutation) columns should be exactly zero before z-scoring, i.e.
    # some columns should have ~zero variance in the raw (unscaled) sense.
    # We check this indirectly: get_batch should simply succeed and produce
    # finite output even at a large num_features.
    rng = np.random.default_rng(2)
    x, y, y_ = module.get_batch(batch_size=2, seq_len=20, num_features=120, rng=rng, max_actions=MAX_ACTIONS)
    assert torch.all(torch.isfinite(x))
    assert torch.all(torch.isfinite(y))


def test_lds_raw_trajectory_has_high_autocorrelation():
    """The defining property the paper attributes to LDS: high temporal
    autocorrelation, in contrast to MDP/Bandit. Checked on the *raw*
    per-dimension trajectory before permutation (permutation doesn't change
    autocorrelation, but testing the raw signal is clearer to read).
    """
    rng = np.random.default_rng(3)
    states, actions = lds._sample_one_task(rng, seq_len=500, num_features=20, max_actions=MAX_ACTIONS, config=lds.CONFIG)

    lag1_autocorr = _mean_lag1_autocorr(states)
    assert lag1_autocorr > 0.8, f"expected strong lag-1 autocorrelation from decay rates near 1, got {lag1_autocorr:.3f}"


def test_bandit_states_are_iid_not_autocorrelated():
    """Contrasts directly with the LDS test above: the paper explicitly
    describes Bandit as removing temporal structure entirely (states are
    i.i.d., not a rollout).
    """
    rng = np.random.default_rng(4)
    x, actions = bandit._sample_one_task(rng, seq_len=500, num_features=20, max_actions=MAX_ACTIONS, config=bandit.CONFIG)
    # x mixes state columns with chosen-action-feature columns, but every
    # column is i.i.d. across timesteps by construction (states are sampled
    # i.i.d., and each timestep's action/noise draw is independent), so
    # autocorrelation should be near zero regardless of which columns we check.
    lag1_autocorr = _mean_lag1_autocorr(x)
    assert abs(lag1_autocorr) < 0.2, f"expected near-zero autocorrelation for i.i.d. bandit states, got {lag1_autocorr:.3f}"


def test_bandit_reward_structure_affects_action_choice():
    """At low temperature the policy should concentrate on the
    highest-reward action far more than chance; at high temperature it
    should be close to uniform. This confirms the reward function actually
    has structure influencing action probabilities, not just noise.
    """
    config_low_temp = {**bandit.CONFIG, "temperature_range": (0.01, 0.01)}
    config_high_temp = {**bandit.CONFIG, "temperature_range": (50.0, 50.0)}

    rng_low = np.random.default_rng(5)
    _, actions_low = bandit._sample_one_task(rng_low, seq_len=300, num_features=20, max_actions=MAX_ACTIONS, config=config_low_temp)
    rng_high = np.random.default_rng(5)
    _, actions_high = bandit._sample_one_task(rng_high, seq_len=300, num_features=20, max_actions=MAX_ACTIONS, config=config_high_temp)

    low_temp_concentration = np.bincount(actions_low.astype(int)).max() / len(actions_low)
    high_temp_concentration = np.bincount(actions_high.astype(int)).max() / len(actions_high)
    assert low_temp_concentration > high_temp_concentration


def test_mdp_transitions_are_valid_and_deterministic():
    rng = np.random.default_rng(6)
    d, n_states, n_actions = 10, 50, 5
    state_coords = rng.uniform(0, 1, size=(n_states, d))

    f_a = mdp.sample_rff_function(rng, d_in=d, d_out=1 + d, n_rff=32, lengthscale=0.5)
    out = f_a(state_coords)
    next_continuous = state_coords + 0.1 * out[:, 1:]
    nearest = mdp._nearest_state(next_continuous, state_coords)

    assert nearest.shape == (n_states,)
    assert np.all(nearest >= 0) and np.all(nearest < n_states)
    # Determinism: same query against the same state set gives the same answer.
    nearest_again = mdp._nearest_state(next_continuous, state_coords)
    np.testing.assert_array_equal(nearest, nearest_again)


def test_mdp_value_iteration_improves_reward_seeking():
    """A hand-built 2-state MDP where action 0 always gives higher reward
    and keeps the agent in the high-reward state: value iteration should
    assign it a strictly higher Q-value, and the resulting Boltzmann policy
    (at low temperature) should prefer it.
    """
    reward = np.array([[1.0, 0.0], [1.0, 0.0]])  # action 0 always better
    transition = np.array([[0, 1], [0, 1]], dtype=np.int64)  # action 0 stays at 0 (high reward)

    V = mdp._truncated_value_iteration(reward, transition, gamma=0.9, n_iters=50)
    Q = reward + 0.9 * V[transition]
    assert Q[0, 0] > Q[0, 1]
    assert Q[1, 0] > Q[1, 1]


def test_mdp_policy_is_not_fully_deterministic_at_default_temperature():
    """With Boltzmann exploration, the rollout should use more than one
    action over a long horizon -- otherwise the generator degenerates into
    a deterministic policy, losing the "vary exploration" property the
    paper attributes to random Boltzmann temperatures.
    """
    rng = np.random.default_rng(7)
    _, actions = mdp._sample_one_task(rng, seq_len=300, num_features=20, max_actions=MAX_ACTIONS, config=mdp.CONFIG)
    assert len(np.unique(actions)) > 1


def _mean_lag1_autocorr(x: np.ndarray) -> float:
    """Mean lag-1 autocorrelation across columns of x: (seq_len, d)."""
    x_centered = x - x.mean(axis=0, keepdims=True)
    num = (x_centered[:-1] * x_centered[1:]).sum(axis=0)
    den = (x_centered**2).sum(axis=0)
    den = np.where(den < 1e-12, 1.0, den)
    return float(np.mean(num / den))
