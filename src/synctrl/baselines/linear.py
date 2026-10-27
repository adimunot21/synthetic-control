"""Multinomial logistic regression baseline, fit fresh per task on the
context. Standardized the same way as the kNN baseline (context-only
statistics), so differences between the two reflect the modeling choice
(local averaging vs. a global linear decision boundary), not preprocessing.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Baseline, expand_to_full_action_space, single_class_proba_or_none, smooth_and_normalize


class LogisticRegressionBaseline(Baseline):
    name = "logistic_regression"

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

        mean = context_x.mean(axis=0)
        std = context_x.std(axis=0)
        std[std < 1e-8] = 1.0
        context_x_std = (context_x - mean) / std
        query_x_std = (query_x - mean) / std

        clf = LogisticRegression(max_iter=1000)
        clf.fit(context_x_std, context_y)
        proba_seen = clf.predict_proba(query_x_std)
        proba = expand_to_full_action_space(proba_seen, clf.classes_, n_classes)
        return smooth_and_normalize(proba, n_classes)
