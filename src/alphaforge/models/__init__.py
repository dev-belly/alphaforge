"""Machine-learning alpha engine: dataset, walk-forward splits, models, metrics."""

from alphaforge.models.dataset import AlphaDataset, build_dataset, to_matrix
from alphaforge.models.estimators import ModelConfig, get_estimator
from alphaforge.models.evaluation import ModelEvaluation, evaluate_predictions
from alphaforge.models.pipeline import (
    AlphaModelPipeline,
    WalkForwardResult,
    run_model_comparison,
    signal_panel,
)
from alphaforge.models.split import WalkForwardConfig, WalkForwardSplitter

__all__ = [
    "AlphaDataset",
    "build_dataset",
    "to_matrix",
    "ModelConfig",
    "get_estimator",
    "WalkForwardSplitter",
    "WalkForwardConfig",
    "AlphaModelPipeline",
    "WalkForwardResult",
    "run_model_comparison",
    "signal_panel",
    "ModelEvaluation",
    "evaluate_predictions",
]
