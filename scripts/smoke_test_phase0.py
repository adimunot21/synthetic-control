"""Phase 0 end-to-end smoke test: sim rollout -> context/query split -> tiny TabPFN forward pass.

Per the global external-interfaces rule: one real sample through every pipeline
stage, printing shapes/types/values, before any heavy work. This does not
train anything -- it only proves the shapes and conventions documented in
docs/data-contract-tabpfn.md actually compose end-to-end on CPU with a tiny
architecture, using one of our own Phase 0 simulators as the data source.
"""

from __future__ import annotations

import numpy as np
import torch
from tabpfn.encoders import Linear as TabPFNLinearEncoder
from tabpfn.transformer import TransformerModel

from synctrl.sims.pendulum import Pendulum

TINY_MODEL_CONFIG = {
    "ninp": 16,  # embedding dim; tiny on purpose, this is a shape smoke test only
    "nhead": 2,
    "nhid": 32,
    "nlayers": 2,
}


def main() -> None:
    print("--- Stage 1: simulate a rollout ---")
    rng = np.random.default_rng(0)
    sim = Pendulum()
    rollout = sim.rollout(n_steps=60, rng=rng, noise_eps=0.1)
    states = rollout.states[:-1]  # state BEFORE each action, (n_steps, state_dim)
    actions = rollout.actions  # (n_steps,) int
    print("states shape/dtype:", states.shape, states.dtype)
    print("actions shape/dtype:", actions.shape, actions.dtype, "unique:", np.unique(actions))

    print("\n--- Stage 2: context/query split (TabPFN seq-first layout) ---")
    n_context = 40
    x = torch.tensor(states, dtype=torch.float32).unsqueeze(1)  # (seq_len, batch=1, num_features)
    y = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)  # (seq_len, batch=1)
    print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
    print(f"single_eval_pos (context size) = {n_context}, query size = {states.shape[0] - n_context}")

    print("\n--- Stage 3: tiny TabPFN forward pass ---")
    n_classes = sim.action_space_size
    model = TransformerModel(
        encoder=TabPFNLinearEncoder(states.shape[1], TINY_MODEL_CONFIG["ninp"]),
        n_out=n_classes,
        ninp=TINY_MODEL_CONFIG["ninp"],
        nhead=TINY_MODEL_CONFIG["nhead"],
        nhid=TINY_MODEL_CONFIG["nhid"],
        nlayers=TINY_MODEL_CONFIG["nlayers"],
        y_encoder=TabPFNLinearEncoder(1, TINY_MODEL_CONFIG["ninp"]),
    )
    model.eval()
    with torch.no_grad():
        logits = model((x, y), single_eval_pos=n_context)
    print("logits shape:", tuple(logits.shape), "dtype:", logits.dtype)

    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5), (
        "predictive distribution does not sum to 1 -- shape/axis contract violated"
    )
    pred_actions = probs.argmax(-1).squeeze(-1).numpy()
    true_actions = actions[n_context:]
    print("predicted action indices:", pred_actions)
    print("true (expert+noise) action indices:", true_actions)
    print(
        "accuracy of an UNTRAINED tiny model (sanity only, not a real result, "
        f"expect roughly chance level): {(pred_actions == true_actions).mean():.3f}"
    )
    print("\nEND-TO-END SMOKE TEST PASSED: sim rollout -> context/query split -> TabPFN forward pass, shapes consistent.")


if __name__ == "__main__":
    main()
