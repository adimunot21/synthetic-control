"""Extends the smoke-scale training run from its latest checkpoint for a
much longer budget, after the initial 800-step run turned out to be
massively undertrained relative to any fair test of RQ1 (see the
conversation / commit history around results/phase1_rq1_decision_gate.json).

Still laptop-only, no cloud spend: ~8000 additional steps at the measured
~1.1s/step is roughly 2.5 hours.
"""

from __future__ import annotations

from pathlib import Path

from synctrl.pretrain import CONFIG, train

EXTEND_STEPS = 8000

if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints")
    latest = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]

    extended_config = {**CONFIG, "n_steps": EXTEND_STEPS}
    print(f"Resuming from {latest}, running {EXTEND_STEPS} additional steps.")
    train(config=extended_config, checkpoint_dir=checkpoint_dir, resume_from=latest)
