"""The control-prior model: a TabPFN-style Transformer trained on our
generator mixture, wrapped to match the `Predictor` protocol the baseline
suite already implements (synctrl.eval.harness.Predictor) -- so it can be
scored with exactly the same harness, metrics, and bootstrap CIs.

Architecture reuses tabpfn.transformer.TransformerModel directly (validated
end-to-end in Phase 0's smoke test), not a reimplementation -- only the
*size* differs between "smoke scale" (this project's laptop-feasible
budget) and the paper's own 12-layer/256-dim/30h-training configuration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tabpfn.encoders import Linear as TabPFNLinearEncoder
from tabpfn.transformer import TransformerModel

from .generators.base import pad_and_permute_features, z_score


def build_model(num_features: int, n_classes: int, ninp: int, nhead: int, nhid: int, nlayers: int) -> TransformerModel:
    return TransformerModel(
        encoder=TabPFNLinearEncoder(num_features, ninp),
        n_out=n_classes,
        ninp=ninp,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        y_encoder=TabPFNLinearEncoder(1, ninp),
    )


class ControlPriorModel:
    """Wraps a trained TransformerModel for per-task in-context prediction
    on arbitrary (context_x, context_y, query_x) inputs whose feature
    dimensionality may be smaller than the model's training-time
    num_features -- padded and permuted the same way the pretraining
    generators are (synctrl.generators.base), since that's the
    distribution the model was actually trained to expect.
    """

    name = "control_prior_model"

    def __init__(self, model: TransformerModel, num_features: int, n_out: int, device: str, rng: np.random.Generator):
        self.model = model
        self.num_features = num_features
        self.n_out = n_out
        self.device = device
        self.rng = rng
        self.model.eval()

    def predict_proba(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        assert n_classes <= self.n_out, (
            f"task has {n_classes} actions but this model was trained with a max of {self.n_out}"
        )
        n_context = context_x.shape[0]

        # x values (unlabeled) are visible to the model at every position,
        # context and query alike -- that's the premise of in-context
        # prediction -- so padding/permuting/z-scoring the concatenated
        # sequence matches exactly how pretraining tasks are constructed
        # (see generators/base.py), not a leakage shortcut.
        combined_x = np.concatenate([context_x, query_x], axis=0).astype(np.float64)
        combined_x = z_score(pad_and_permute_features(combined_x, self.num_features, self.rng))

        x = torch.tensor(combined_x, dtype=torch.float32, device=self.device).unsqueeze(1)  # (seq_len, 1, num_features)
        y_context = torch.tensor(context_y, dtype=torch.float32, device=self.device).unsqueeze(1)  # (n_context, 1)
        # Query-position labels are never read by the model (TransformerModel
        # masks them out via single_eval_pos), but the tensor still needs a
        # placeholder row per query position to keep shapes consistent.
        y_placeholder = torch.zeros(x.shape[0] - n_context, 1, device=self.device)
        y = torch.cat([y_context, y_placeholder], dim=0)

        with torch.no_grad():
            logits = self.model((x, y), single_eval_pos=n_context)  # (n_query, 1, n_out)
        full_proba = torch.softmax(logits, dim=-1).squeeze(1).cpu().numpy()  # (n_query, n_out)

        proba = full_proba[:, :n_classes]
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: (n_query, batch, n_out), targets: (n_query, batch) float
    labels (cast to long for the loss)."""
    n_query, batch, n_out = logits.shape
    return nn.functional.cross_entropy(logits.reshape(n_query * batch, n_out), targets.reshape(-1).long())
