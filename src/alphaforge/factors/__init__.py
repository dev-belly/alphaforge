"""Cross-sectional factor research engine."""

from alphaforge.factors.base import (
    CATEGORIES,
    REGISTRY,
    Factor,
    FactorContext,
    FactorSpec,
    FactorUnavailableError,
)
from alphaforge.factors.evaluation import FactorResult, evaluate_factor, factor_correlation
from alphaforge.factors.library import FactorLibrary
from alphaforge.factors.preprocessing import FactorPreprocessor, ProcessingConfig

__all__ = [
    "FactorSpec",
    "Factor",
    "FactorContext",
    "FactorUnavailableError",
    "FactorLibrary",
    "FactorPreprocessor",
    "ProcessingConfig",
    "FactorResult",
    "evaluate_factor",
    "factor_correlation",
    "REGISTRY",
    "CATEGORIES",
]
