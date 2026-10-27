import numpy as np
import torch

from synctrl.model import ControlPriorModel, build_model
from synctrl.sims.pendulum import Pendulum
from synctrl.tasks import sample_tasks


def test_build_model_forward_shapes():
    model = build_model(num_features=10, n_classes=6, ninp=16, nhead=2, nhid=32, nlayers=2)
    seq_len, batch, num_features = 30, 2, 10
    x = torch.randn(seq_len, batch, num_features)
    y = torch.randint(0, 6, (seq_len, batch)).float()

    model.eval()
    with torch.no_grad():
        logits = model((x, y), single_eval_pos=20)
    assert logits.shape == (10, batch, 6)


def test_control_prior_model_predict_proba_on_sim_task():
    model = build_model(num_features=10, n_classes=6, ninp=16, nhead=2, nhid=32, nlayers=2)
    wrapped = ControlPriorModel(model, num_features=10, n_out=6, device="cpu", rng=np.random.default_rng(0))

    rng = np.random.default_rng(1)
    system = Pendulum()
    task = sample_tasks(system, n_tasks=1, n_steps=40, rng=rng, noise_eps=0.1, split="contiguous")[0]

    proba = wrapped.predict_proba(task.context_x, task.context_y, task.query_x, task.action_space_size)
    assert proba.shape == (task.query_x.shape[0], task.action_space_size)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(proba >= 0)


def test_control_prior_model_handles_n_classes_less_than_n_out():
    # n_out=10 (trained with max_actions=10), task only has 3 actions --
    # the model must restrict+renormalize, not error or return 10 columns.
    model = build_model(num_features=8, n_classes=10, ninp=16, nhead=2, nhid=32, nlayers=2)
    wrapped = ControlPriorModel(model, num_features=8, n_out=10, device="cpu", rng=np.random.default_rng(0))

    context_x = np.random.default_rng(2).normal(size=(20, 3))
    context_y = np.random.default_rng(3).integers(0, 3, size=20).astype(np.float64)
    query_x = np.random.default_rng(4).normal(size=(5, 3))

    proba = wrapped.predict_proba(context_x, context_y, query_x, n_classes=3)
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
