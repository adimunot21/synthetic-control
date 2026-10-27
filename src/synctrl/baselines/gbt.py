"""Gradient-boosted-trees baseline, fit fresh per task on the context.
Captures nonlinear, non-smooth decision boundaries that linear/kNN can't,
at the cost of needing more context to avoid overfitting a tiny per-task
training set -- that tradeoff is itself informative about how much context
each kind of structure needs.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import Baseline, expand_to_full_action_space, single_class_proba_or_none, smooth_and_normalize


class GradientBoostedTreesBaseline(Baseline):
    name = "gbt"

    def predict_proba(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        fallback = single_class_proba_or_none(context_y, query_x.shape[0], n_classes)
        if fallback is not None:
            return smooth_and_normalize(fallback, n_classes)

        # Small, fast settings: per-task fits need to run in milliseconds
        # across thousands of tasks, and context windows here are at most a
        # few hundred points, too small to need deep/wide boosting.
        clf = HistGradientBoostingClassifier(max_iter=50, max_depth=4)
        clf.fit(context_x, context_y)
        proba_seen = clf.predict_proba(query_x)
        proba = expand_to_full_action_space(proba_seen, clf.classes_, n_classes)
        return smooth_and_normalize(proba, n_classes)
