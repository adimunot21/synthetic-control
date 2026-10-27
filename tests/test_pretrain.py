import numpy as np

from synctrl.pretrain import GENERATORS, sample_batch, train

TINY_CONFIG = {
    "ninp": 8,
    "nhead": 2,
    "nhid": 16,
    "nlayers": 1,
    "num_features": 10,
    "max_actions": 4,
    "seq_len": 20,
    "single_eval_pos": 12,
    "batch_size": 4,
    "n_steps": 3,
    "lr": 1e-3,
    "log_every": 1,
    "checkpoint_every": 3,
    "seed": 0,
}


def test_sample_batch_shapes():
    rng = np.random.default_rng(0)
    x, y, generator_name = sample_batch(TINY_CONFIG, rng)
    assert x.shape == (TINY_CONFIG["seq_len"], TINY_CONFIG["batch_size"], TINY_CONFIG["num_features"])
    assert y.shape == (TINY_CONFIG["seq_len"], TINY_CONFIG["batch_size"])
    assert generator_name in GENERATORS


def test_sample_batch_uses_all_generators_over_many_calls():
    rng = np.random.default_rng(0)
    seen = {sample_batch(TINY_CONFIG, rng)[2] for _ in range(50)}
    assert seen == set(GENERATORS.keys())


def test_train_runs_and_checkpoints(tmp_path):
    result = train(config=TINY_CONFIG, checkpoint_dir=tmp_path)
    assert len(result["history"]) == TINY_CONFIG["n_steps"]
    assert all(np.isfinite(r["loss"]) for r in result["history"])

    checkpoint_files = list(tmp_path.glob("step_*.pt"))
    assert len(checkpoint_files) == 1  # checkpoint_every == n_steps here
    assert (tmp_path / "history.json").exists()
