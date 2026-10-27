"""Majority-class baseline: predicts the context's most frequent action for
every query point, regardless of state. This is the floor any "the model
learned structure" claim must clear -- if a method doesn't beat this, it
isn't using the query state at all.
"""

from __future__ import annotations

import numpy as np

from .base import Baseline, smooth_and_normalize


class MajorityClassBaseline(Baseline):
    name = "majority"

    def predict_proba(
        self,
        context_x: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        n_query = query_x.shape[0]
        counts = np.bincount(context_y, minlength=n_classes).astype(np.float64)
        proba_row = counts / counts.sum() if counts.sum() > 0 else np.ones(n_classes) / n_classes
        proba = np.tile(proba_row, (n_query, 1))
        return smooth_and_normalize(proba, n_classes)
