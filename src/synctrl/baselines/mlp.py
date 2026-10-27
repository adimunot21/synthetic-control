"""Small MLP baseline, fit fresh per task on the context with a short,
fixed training budget. Not meant to be competitive with a pretrained
in-context model on its own terms (a from-scratch net fit on ~50-500 points
in a handful of gradient steps is a weak learner by design) -- it's here as
a nonlinear complement to logistic regression and GBT, not a serious
contender.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from .base import Baseline, expand_to_full_action_space, single_class_proba_or_none, smooth_and_normalize


class SmallMLPBaseline(Baseline):
    name = "small_mlp"

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

        # A short, fixed budget is deliberate: this baseline should reflect
        # what a small net can learn from one task's context alone, not a
        # fully-converged fit that would take far longer than the per-task
        # budget any realistic baseline comparison allows.
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, early_stopping=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(context_x_std, context_y)
        proba_seen = clf.predict_proba(query_x_std)
        proba = expand_to_full_action_space(proba_seen, clf.classes_, n_classes)
        return smooth_and_normalize(proba, n_classes)
