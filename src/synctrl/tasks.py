"""The common Task interface eval baselines and models are compared on.

A Task is one (context, query) action-prediction problem, matching the
paper's protocol (Section III): a model or baseline sees `context_x` paired
with `context_y`, and must predict `query_y` from `query_x` alone.

Two split protocols, per docs/00_critique_and_plan.md's leakage-control plan:

- "contiguous": context = the first half of the window in time, query = the
  second half. Context strictly precedes query. This is the realistic
  protocol -- predicting from the past, not interpolating within a shuffled
  window.
- "random": all timesteps are shuffled before the context/query split, so
  context and query points are randomly interleaved in time. This is a
  *deliberate leakage control*, not a realistic eval protocol: query points
  can now be temporally adjacent to context points purely by chance, so a
  method that does much better here than under "contiguous" is likely
  exploiting local temporal proximity rather than genuinely extrapolating.
  The gap between the two is itself a diagnostic for the project's central
  skeptical hypothesis (local smoothing vs. learned control structure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sims.base import KnownExpertSystem, Rollout


@dataclass
class Task:
    context_x: np.ndarray  # (n_context, feature_dim)
    context_y: np.ndarray  # (n_context,) int labels
    query_x: np.ndarray  # (n_query, feature_dim)
    query_y: np.ndarray  # (n_query,) int labels, ground truth to score against
    action_space_size: int
    split: str  # "contiguous" or "random", for bookkeeping in results
    system_params: dict  # the sampled physical parameters for this task's rollout


def rollout_to_task(
    system: KnownExpertSystem,
    rollout: Rollout,
    split: str,
    rng: np.random.Generator,
    context_frac: float = 0.5,
) -> Task:
    if split not in ("contiguous", "random"):
        raise ValueError(f"split must be 'contiguous' or 'random', got {split!r}")

    n_steps = len(rollout.actions)
    n_context = int(round(n_steps * context_frac))

    # states[:-1] is the state *before* each action -- states[t] paired with
    # actions[t], consistent with how the known-expert sims generate data.
    x_all = rollout.states[:-1]
    y_all = rollout.actions

    if split == "contiguous":
        order = np.arange(n_steps)
    else:
        order = rng.permutation(n_steps)

    context_idx = order[:n_context]
    query_idx = order[n_context:]

    return Task(
        context_x=x_all[context_idx],
        context_y=y_all[context_idx],
        query_x=x_all[query_idx],
        query_y=y_all[query_idx],
        action_space_size=system.action_space_size,
        split=split,
        system_params=rollout.params,
    )


def sample_tasks(
    system: KnownExpertSystem,
    n_tasks: int,
    n_steps: int,
    rng: np.random.Generator,
    noise_eps: float,
    split: str,
    context_frac: float = 0.5,
) -> list[Task]:
    """One rollout per task, independently parameterized (fresh randomized
    physical parameters each time), matching the "hold out whole system
    families" spirit: every task is a different instance of the system.
    """
    tasks = []
    for _ in range(n_tasks):
        rollout = system.rollout(n_steps=n_steps, rng=rng, noise_eps=noise_eps)
        tasks.append(rollout_to_task(system, rollout, split=split, rng=rng, context_frac=context_frac))
    return tasks
