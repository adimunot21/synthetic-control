"""k-nearest-neighbors baseline: the direct test of the project's central
skeptical hypothesis (docs/00_critique_and_plan.md §3.1) -- that in-context
"control" performance is largely local smoothing/interpolation over an
autocorrelated state-action stream, which kNN implements almost literally.

Features are standardized using context-only statistics (never query
statistics) before computing distances, mirroring the paper's own z-scoring
of generator features (Section IV) and avoiding any leakage from the query
set into the distance metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .base import Baseline, expand_to_full_action_space, single_class_proba_or_none, smooth_and_normalize


class KNNBaseline(Baseline):
    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    def predict_proba(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        mean = context_x.mean(axis=0)
        std = context_x.std(axis=0)
        std[std < 1e-8] = 1.0  # guard against a constant feature column
        context_x_std = (context_x - mean) / std
        query_x_std = (query_x - mean) / std

        fallback = single_class_proba_or_none(context_y, query_x.shape[0], n_classes)
        if fallback is not None:
            return smooth_and_normalize(fallback, n_classes)

        k_eff = min(self.k, context_x.shape[0])
        clf = KNeighborsClassifier(n_neighbors=k_eff)
        clf.fit(context_x_std, context_y)
        proba_seen = clf.predict_proba(query_x_std)
        proba = expand_to_full_action_space(proba_seen, clf.classes_, n_classes)
        return smooth_and_normalize(proba, n_classes)
