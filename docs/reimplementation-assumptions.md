# Reimplementation Assumptions Log

Per docs/00_critique_and_plan.md's risk register: the paper's §IV generator descriptions specify ranges and high-level mechanisms, but not every implementation detail (exact RFF feature count, value-iteration discount/horizon, etc.). This file logs every place we filled in a gap, so results are never silently presented as an exact reproduction. Each entry: what's unspecified, what we chose, and why.

## Shared across all generators

- **RFF-based GP sampling**: the paper says reward/transition functions are "jointly sampled from a multi-output Gaussian-process prior (via random Fourier features)" (MDP) and reward is "a smooth bilinear form... over random-Fourier-feature embeddings" (Bandit). We implement the standard RFF approximation to a GP sample: `f(x) = sum_i w_i * cos(omega_i . x + phi_i)`, with `omega_i ~ N(0, 1/lengthscale^2)` (independent per output dimension unless otherwise noted), `phi_i ~ Uniform(0, 2*pi)`, `w_i ~ N(0, 1)` scaled by `1/sqrt(n_rff)`. Number of RFF features `n_rff` and the lengthscale are not given in the paper; chosen as `n_rff=64`, `lengthscale` sampled per-task from a broad log-uniform range (see each generator's `CONFIG`) so smoothness itself varies across sampled tasks, matching the spirit of "random Fourier features" without a fixed single bandwidth.
- **Feature padding/permutation**: the paper permutes feature dimensions per task so "no input index carries consistent meaning." Our generators sample an *intrinsic* dimensionality `d` within the paper's stated range, zero-pad up to the caller-requested `num_features` if `d < num_features`, then permute all `num_features` columns together. Not stated whether padding is zeros, noise, or omitted entirely (i.e. `num_features` always equals `d`); we chose zero-padding as the simplest, most neutral choice, flagged here as a filled gap.
- **Z-scoring**: §IV states every sampled task's features are z-scored "to ensure scale consistency." Applied identically across all three generators as a shared post-processing step (mean/std computed over the sampled task's own context+query pool, not leaking across tasks).

## MDP (`generators/mdp.py`)

- **Value iteration discount factor and horizon**: not stated. Chose `gamma=0.95`, truncated to `n_value_iterations=50` backward passes (the paper says "truncated backward value iteration" but not how many iterations). Both are CONFIG entries so this is easy to revisit in an ablation if the chosen values turn out to matter.
- **"Actions acting as small spatial displacements"**: read as describing the *effect scale* of the single GP-sampled transition function, not a separate displacement parameterization layered on top of it. Concretely: one multi-output RFF-GP function `f_a: R^d -> R^{1+d}` is sampled per action (shared basis across its 1+d outputs, matching "jointly sampled... reward and transition"); the first output is `reward(s,a)`, the remaining `d` outputs are scaled by a small constant (`displacement_scale` in CONFIG, a fraction of the unit cube) and added to `s` to get the continuous next-state before nearest-state snapping. This keeps reward and transition as one unified GP sample exactly as the paper's sentence states, with "small spatial displacement" emerging from the scale constant rather than a second, redundant per-action vector.
- **"Snapped to nearest discrete state"**: implemented as exact nearest-neighbor (Euclidean) lookup into the sampled state set after applying the continuous RFF-sampled transition function plus the action's displacement.

## Contextual Bandit (`generators/bandit.py`)

- **`num_features` floor**: the generator concatenates state features and chosen-action features into one input row, so `num_features` must be at least `state_dim_min + action_feat_dim_min` (4 + 4 = 8 with the ranges below); `get_batch` asserts this rather than silently truncating. Not a paper detail, just a consequence of fitting two sampled dimensions into one caller-specified feature budget.

- **"Bilinear form of random rank"**: implemented as `f(s, a) = phi_s(s)^T @ M @ phi_a(a)`, where `phi_s`, `phi_a` are RFF embeddings of the state and the action's feature vector respectively, and `M` is a random low-rank matrix (`rank` sampled per-task within a chosen range, since the paper doesn't give one) — this is the standard meaning of "bilinear form of random rank" and is a direct, low-assumption reading of the paper's text, not a free design choice beyond picking the rank's sampling range.
- **Action feature perturbation noise scale**: paper says the chosen action's features are "perturbed by Gaussian noise so they do not identify the chosen action" but doesn't give a noise scale; chosen as a fraction of the action feature vectors' own empirical std within the task.

## Pretraining loop (`pretrain.py`)

- **Generator sampled per batch, not per task**: the paper draws a generator `g ~ Uniform(G)` independently for *every task* (Section IV). For vectorization/speed we draw one generator for an entire batch of tasks instead (still uniform over the three generators, still re-drawn every training step), since each of our generators' `get_batch` already independently re-samples a fresh system per task within the batch. Over many training steps this converges to the same per-generator exposure rate; it only changes the *granularity* of mixing, not the mixture itself. Flagged here since it's a real simplification, not a hidden one.

## LDS (`generators/lds.py`)

- **Decay-rate distribution ("near one")**: not quantified. Chosen as `Uniform(0.9, 0.999)` per diagonal entry of `A`, independently per state dimension — high enough to guarantee strong autocorrelation (the property the paper explicitly attributes to this generator) while avoiding instability (`>1`) or triviality (`<<1`, which would wash out autocorrelation and defeat the generator's purpose).
- **Policy class ("linear-softmax")**: implemented as `pi(a|s) = softmax(W_a . s + b_a)` with `W_a`, `b_a` sampled per-task, matching the literal name; no further detail was given to deviate from.
