# Data Contract: `tabpfn==0.1.11`

Status: **Phase 0 inspection complete.** Everything below was either read directly from the installed package source or produced by an actual call in this environment — nothing here is assumed.

Per the global external-interfaces rule: inspect before integrating, don't assume semantics, document the contract, smoke-test end-to-end before scaling.

## Source & install

- PyPI `tabpfn==0.1.11` — the ICLR 2023 TabPFN v1 codebase (see [docs/00_critique_and_plan.md](00_critique_and_plan.md) §4 for why this version specifically, and the license note).
- Base install (`pip install tabpfn==0.1.11 gpytorch`) needs only numpy, pyyaml, requests, scikit-learn, torch>=1.9.0, plus `gpytorch` for the GP prior. The `[full]` extras (auto-sklearn, catboost, hyperopt, openml — original paper's AutoML baselines, not needed here) were still backtracking dependency resolution after 5+ minutes when tested; avoided entirely.
- `ConfigSpace` is a separate, lightweight additional dependency needed only to build default hyperparameter *distributions* via `tabpfn.scripts.model_configs` (unrelated to the AutoML baselines; installed standalone, `configspace==1.2.0`).

## Required patch: torch compatibility

`tabpfn/layer.py` does:

```python
from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, Optional, MultiheadAttention, Linear, Dropout, LayerNorm
```

`Optional` was never actually defined in that torch module — it was only reachable there in ~2022-era torch because `torch.nn.modules.transformer` did a `from typing import *`-style import at the time. Modern torch (tested: 2.14.1+cu130) imports `typing` names explicitly and no longer re-exports `Optional`, so this import now raises `ImportError`. Every other name in that line still resolves correctly from the current torch module (checked directly before patching).

Fix applied: [scripts/patch_tabpfn_torch_compat.py](../scripts/patch_tabpfn_torch_compat.py) — rewrites the one import line to pull `Optional` from `typing`. Idempotent; run once after any fresh `pip install tabpfn==0.1.11` in a new environment.

## Known upstream bug (not ours to fix, just documented)

`tabpfn/scripts/model_configs.py::get_prior_config_gp()` is missing its `return config` statement — it builds the config dict, mutates one entry, and falls off the end of the function, so `get_prior_config(config_type='gp')` returns `None` and raises `TypeError: argument of type 'NoneType' is not iterable` downstream. `get_prior_config_causal()` and (presumably — not yet checked) `get_prior_config_bnn()` do return correctly. **Workaround for Phase 1: don't rely on `get_prior_config('gp')`; either call `fast_gp.get_batch` directly with a hand-built hyperparameters dict (see below), or patch this function the same way as the torch-compat fix if the full differentiable-hyperparameter config is needed later.**

## Architecture defaults (resolves the "unknown parameter count" item from the critique doc)

From `get_prior_config(config_type='causal')`'s architecture-relevant keys, read directly:

| key | value |
|---|---|
| `emsize` | 256 |
| `nlayers` | 12 |
| `nhead` | 4 |
| `nhid_factor` | 2 (→ feedforward dim 512) |
| `batch_size` (default) | 64 |
| `bptt` (default) | 50 |

This confirms the paper's "12-layer Transformer with 4 attention heads" (§VI-D) is the library's unmodified default architecture — the paper doesn't appear to have changed depth/width, only dataset size (2048, up from 1024) and training budget (dataset count / epochs). Exact parameter count not yet computed from `nn.Module` directly (trivial to do in Phase 1 with `sum(p.numel() for p in model.parameters())` once we instantiate one). **[estimated, not computed]**: a 12-layer/256-dim/4-head/512-hidden encoder is roughly in the single-digit-millions-of-parameters range by the standard transformer-encoder parameter formula — small, consistent with the paper's own scale caveat (§VI-B), but treat this as a back-of-envelope estimate until Phase 1 actually instantiates the model and counts.

## Generator composition (from `tabpfn/scripts/model_builder.py::get_model`)

Confirms and sharpens the paper's §IV-A description ("SCM/BNN/GP mixture"):

- **BNN** ≈ `tabpfn.priors.mlp.get_batch` (a random MLP prior).
- **GP** ≈ `tabpfn.priors.fast_gp.get_batch`.
- **SCM** ≈ either of the above *wrapped* in `tabpfn.priors.flexible_categorical.get_batch` (via its `get_batch=...` kwarg), which adds categorical/causal-structure post-processing on top of a base continuous generator.
- **The "generic tabular prior"** = `tabpfn.priors.prior_bag.get_batch`, which samples per-task from a weighted mixture of the above (`prior_bag_get_batch=(get_batch_gp, get_batch_mlp)`, with a sampled mixture weight — default setup weights the MLP prior higher, "MLP works better" per an inline code comment).

This is useful for Phase 1 but is **not** the generator we're reimplementing for our own MDP/Bandit/LDS pipelines — those are our own code per the paper's §IV-B description. We only need this `tabpfn.priors.*` machinery to **reuse the tabular prior as a baseline**, exactly as the paper does.

## Real sample: `get_batch` I/O contract

Direct call (hand-built hyperparameters, bypassing the broken/complex config-generation layer entirely — sufficient for inspecting the contract):

```python
from tabpfn.priors import fast_gp
hp = {"lengthscale": 1.0, "outputscale": 1.0, "noise": 0.1}
x, y, y_ = fast_gp.get_batch(batch_size=4, seq_len=20, num_features=6, hyperparameters=hp, device='cpu')
```

Printed output:

| tensor | shape | dtype | range (this sample) |
|---|---|---|---|
| `x` | `(seq_len, batch_size, num_features)` = `(20, 4, 6)` | `float32` | `[0.0027, 0.9992]` |
| `y` | `(seq_len, batch_size)` = `(20, 4)` | `float32` | `[-2.29, 1.90]` |
| `y_` | `(seq_len, batch_size)` = `(20, 4)` | `float32` | identical to `y` for this generator |

**Contract details worth flagging:**

1. **Axis order is sequence-first, not batch-first**: `(seq_len, batch_size, num_features)`. Easy to get backwards; our own generator code and any wrapper around `tabpfn`'s transformer must match this.
2. **`x` is uniform in `[0, 1]` for the GP prior** (sampled input locations), not standard-normal — don't assume Gaussian inputs when z-scoring/permuting features for our own generators to match.
3. **`y` and `y_` are separate return values that happen to be identical here** — for the GP prior there's no discretization step, so both are the raw continuous GP sample. For `flexible_categorical` (which implements the SCM-style causal prior with categorical binning), these are expected to differ: `y` is the discretized/categorical label, `y_` the pre-discretization continuous value. This is directly relevant to our own K-means discretization + Relative Centroid Distance (RCD) metric work in Phase 1 — `y_` is exactly the kind of "continuous ground truth behind the discrete label" we need to track ourselves for RCD, so the same `(y, y_)` pairing convention is worth mirroring in our own generators rather than inventing a new one.
4. **Not yet resolved**: getting a working `get_batch` call through the full `mlp`/`flexible_categorical` (causal/BNN) path via the library's own differentiable-hyperparameter config system (`sample_differentiable` + `get_mlp_prior_hyperparameters`) failed — some hyperparameters (e.g. `num_layers`) arrive at the `MLP.__init__` as unresolved callables rather than the sampled scalars `sample_differentiable` is supposed to produce, and the exact resolution order used internally by `model_builder.get_model()` → `train()` is more involved than the two-step chain tried here. **Deferred to Phase 1**, flagged explicitly rather than worked around blindly, since this is exactly the kind of "don't assume, inspect further" situation the external-interfaces rule warns about. The GP prior alone is sufficient to validate the I/O contract for Phase 0's purposes.
5. **Inference-time preprocessing** (the PyPI page's warning about internal z-scoring/PowerTransform) applies to the `TabPFNClassifier` sklearn-style wrapper, not to the raw `priors.*.get_batch` functions used here. [verified]: grepped `tabpfn/scripts/transformer_prediction_interface.py` directly — it imports `PowerTransformer`/`QuantileTransformer`/`RobustScaler` from sklearn and `normalize_data`/`to_ranking_low_mem`/`remove_outliers` from `tabpfn.utils`, and has a `no_preprocess_mode` flag on the classifier constructor; none of this appears in `tabpfn/priors/*.py`, which only ever returns the raw sampled `(x, y, y_)`.
