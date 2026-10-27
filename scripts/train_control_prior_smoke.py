"""Entry point for the Phase 1 smoke-scale control-prior pretraining run.
See synctrl/pretrain.py for the CONFIG and training loop itself -- this is
just the runnable wrapper, consistent with the other scripts/ entries.
"""

from synctrl.pretrain import CONFIG, train

if __name__ == "__main__":
    print("Training control-prior model (smoke scale) with config:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()
    train()
