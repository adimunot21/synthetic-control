"""Common interface for per-task baselines.

Every baseline fits on a task's context (x, y) and predicts a full
probability distribution over the task's action space for the query points
-- the same information a pretrained in-context model gets, nothing more.
There is no state carried between tasks: each `predict_proba` call is an
independent fit, matching the in-context-learning setting these baselines
are meant to stress-test (per docs/00_critique_and_plan.md RQ1).
"""

from __future__ import annotations

import abc

import numpy as np

# Additive (Laplace-style) smoothing applied to every baseline's output
# distribution before scoring NLL. Without this, a baseline that assigns
# exactly zero probability to the true query label (e.g. a class never seen
# in context) would produce NLL = +inf and dominate any aggregate metric --
# an artifact of the scoring rule, not a meaningful result. 1e-3 is small
# enough to leave accuracy/argmax unaffected while keeping NLL finite and
# comparable across baselines and the pretrained model (which never emits
# exact zeros either, since it is a softmax).
PROBA_SMOOTHING_EPS = 1e-3


class Baseline(abc.ABC):
    name: str

    @abc.abstractmethod
    def predict_proba(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """Return (n_query, n_classes) probabilities, smoothed and normalized."""


def smooth_and_normalize(proba: np.ndarray, n_classes: int, eps: float = PROBA_SMOOTHING_EPS) -> np.ndarray:
    assert proba.shape[1] == n_classes
    smoothed = proba + eps
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def single_class_proba_or_none(
    context_y: np.ndarray, n_query: int, n_classes: int
) -> np.ndarray | None:
    """If the context contains only one class, sklearn classifiers either
    raise or behave oddly (no decision boundary to fit). Handle that case
    uniformly: predict that class with certainty (smoothed by the caller).
    Returns None when the context has >=2 classes, so the caller proceeds
    with its normal classifier.
    """
    classes_seen = np.unique(context_y)
    if len(classes_seen) > 1:
        return None
    proba = np.zeros((n_query, n_classes))
    proba[:, int(classes_seen[0])] = 1.0
    return proba


def expand_to_full_action_space(
    proba_seen: np.ndarray,
    classes_seen: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Remap a classifier's `predict_proba` (columns = `classes_`, which may
    be a strict subset of the task's full action space if some actions never
    appeared in the context) to a dense (n_query, n_classes) matrix, with
    zero probability for actions that weren't observed. Smoothing is applied
    afterward by the caller, so those zeros don't cause +inf NLL.
    """
    n_query = proba_seen.shape[0]
    full = np.zeros((n_query, n_classes), dtype=np.float64)
    for col, cls in enumerate(classes_seen):
        full[:, int(cls)] = proba_seen[:, col]
    return full
