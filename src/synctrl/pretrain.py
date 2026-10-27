"""Pretraining loop for the control-prior model: samples tasks from a
uniform mixture of the MDP, Bandit, and LDS generators (the paper's
"Control Priors" configuration, Section V-B -- no tabular prior mixed in),
trains a small TabPFN-style Transformer via in-context next-action
prediction, and checkpoints progress.

This is explicitly the "fast smoke-scale" budget from
docs/00_critique_and_plan.md Phase 1, not the paper's own ~30h/15M-dataset
budget -- enough to validate the harness and get a first real signal on
whether pretraining beats the baseline suite (results/phase1_baseline_suite.json)
on our held-out simulated systems. Scaling up is explicitly gated behind
this working end-to-end (Phase 3), with a cost/time estimate before any
cloud spend.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .generators import bandit, lds, mdp
from .model import build_model, cross_entropy_loss

CONFIG = {
    # Architecture: far smaller than the paper's 12-layer/256-dim (Phase 0
    # confirmed those defaults) -- sized to train in minutes, not ~30h, on a
    # single laptop GPU (GTX 1650, 4GB VRAM).
    "ninp": 64,
    "nhead": 4,
    "nhid": 128,
    "nlayers": 4,
    # Data: num_features=20 comfortably covers our eval sims' native state
    # dims (2-4) once padded, while still giving the MDP/Bandit/LDS
    # generators (whose sampled intrinsic dims can be much larger per the
    # paper's ranges) something nontrivial to pad/permute down to -- the
    # point of the permutation test (no input index means anything fixed).
    "num_features": 20,
    # max_actions=10 matches the paper's VMAS-scale cap, and is already
    # well above our eval sims' 2-3 native actions -- the model has to
    # learn to suppress the unused upper action slots per task, exactly as
    # in the paper's variable-action-count design.
    "max_actions": 10,
    "seq_len": 200,  # matches the 100-context/100-query split used for the baseline suite
    "single_eval_pos": 100,
    # Measured on this laptop (CPU generator sampling, not the model, is the
    # bottleneck: the per-task rollout loops in mdp.py/lds.py are pure
    # Python): batch_size=32 costs ~2.5s/step regardless of generator.
    # batch_size=16 + n_steps=800 keeps this run to roughly 15-20 minutes --
    # a genuine smoke-scale budget, not the paper's ~30h/model.
    "batch_size": 16,
    "n_steps": 800,  # smoke-scale budget; see module docstring
    "lr": 1e-4,
    "log_every": 50,
    "checkpoint_every": 500,
    "seed": 0,
}

GENERATORS = {"mdp": mdp, "bandit": bandit, "lds": lds}


def sample_batch(
    config: dict, rng: np.random.Generator, generator_names: tuple[str, ...] = tuple(GENERATORS.keys())
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """One generator drawn uniformly for the whole batch (see
    docs/reimplementation-assumptions.md for why this differs slightly from
    the paper's per-task draw).
    """
    name = generator_names[rng.integers(len(generator_names))]
    x, y, _ = GENERATORS[name].get_batch(
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        num_features=config["num_features"],
        rng=rng,
        max_actions=config["max_actions"],
    )
    return x, y, name


def train(
    config: dict = CONFIG,
    checkpoint_dir: str | Path = "checkpoints",
    resume_from: str | Path | None = None,
) -> dict:
    """If `resume_from` is given, loads that checkpoint's model/optimizer
    state and continues step counting from where it left off -- config's
    `n_steps` is then "how many *additional* steps to run," not a new
    total, so an interrupted or deliberately-staged run can be extended
    without re-training from scratch or losing step-number continuity in
    checkpoint filenames/history.
    """
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        num_features=config["num_features"],
        n_classes=config["max_actions"],
        ninp=config["ninp"],
        nhead=config["nhead"],
        nhid=config["nhid"],
        nlayers=config["nlayers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    start_step = 0
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = payload["step"]
        history_path = checkpoint_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        print(f"Resumed from {resume_from} at step {start_step}; running {config['n_steps']} more steps.")

    end_step = start_step + config["n_steps"]
    start_time = time.time()
    for step in range(start_step + 1, end_step + 1):
        x, y, generator_name = sample_batch(config, rng)
        x, y = x.to(device), y.to(device)

        logits = model((x, y), single_eval_pos=config["single_eval_pos"])
        targets = y[config["single_eval_pos"] :]
        loss = cross_entropy_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config["log_every"] == 0 or step == start_step + 1:
            elapsed = time.time() - start_time
            accuracy = (logits.argmax(dim=-1) == targets.long()).float().mean().item()
            record = {
                "step": step,
                "loss": loss.item(),
                "train_accuracy": accuracy,
                "generator": generator_name,
                "elapsed_s": elapsed,
            }
            history.append(record)
            print(
                f"step {step:5d}/{end_step} | loss {loss.item():.4f} | "
                f"train_acc {accuracy:.3f} | generator {generator_name:7s} | elapsed {elapsed:6.1f}s"
            )

        if step % config["checkpoint_every"] == 0 or step == end_step:
            _save_checkpoint(checkpoint_dir / f"step_{step}.pt", model, optimizer, config, step)

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"model": model, "device": device, "history": history}


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: dict, step: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "step": step,
        },
        path,
    )


if __name__ == "__main__":
    train()
