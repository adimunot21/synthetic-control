from .base import Baseline
from .gbt import GradientBoostedTreesBaseline
from .knn import KNNBaseline
from .linear import LogisticRegressionBaseline
from .majority import MajorityClassBaseline
from .mlp import SmallMLPBaseline

ALL_BASELINES: list[Baseline] = [
    MajorityClassBaseline(),
    KNNBaseline(k=5),
    LogisticRegressionBaseline(),
    GradientBoostedTreesBaseline(),
    SmallMLPBaseline(),
]

__all__ = [
    "Baseline",
    "MajorityClassBaseline",
    "KNNBaseline",
    "LogisticRegressionBaseline",
    "GradientBoostedTreesBaseline",
    "SmallMLPBaseline",
    "ALL_BASELINES",
]
