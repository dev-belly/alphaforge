"""Regression / gradient-boosting estimators for cross-sectional alpha.

The estimators are deliberately plain: the research value is in the
*validation protocol*, not in model novelty.  Every estimator exposes the same
``fit`` / ``predict`` contract so the walk-forward engine can treat them
interchangeably, and every estimator is seeded for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge

from alphaforge.utils.logging import get_logger

log = get_logger("models.estimators")


@dataclass
class ModelConfig:
    type: str = "ridge"
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    @classmethod
    def from_dict(cls, cfg: dict) -> ModelConfig:
        cfg = dict(cfg or {})
        return cls(
            type=str(cfg.get("type", "ridge")).lower(),
            params=dict(cfg.get("params", {}) or {}),
            seed=int(cfg.get("seed", 42)),
        )


class BaseEstimator:
    """Thin wrapper giving every model a uniform interface."""

    name = "base"

    def __init__(self, params: dict | None = None, seed: int = 42) -> None:
        self.params = dict(params or {})
        self.seed = seed
        self.model: Any = None
        self.feature_names: list[str] = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None
    ) -> BaseEstimator:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Estimator has not been fitted")
        return np.asarray(self.model.predict(X), dtype=float)

    def feature_importance(self) -> pd.Series | None:
        return None

    def describe(self) -> dict:
        return {"model": self.name, "params": self.params, "seed": self.seed}


class RidgeModel(BaseEstimator):
    name = "ridge"

    def fit(self, X, y, feature_names=None):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        alpha = float(self.params.get("alpha", 1.0))
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=None))
        self.model.fit(X, y)
        self.feature_names = list(feature_names or [])
        return self

    def feature_importance(self) -> pd.Series | None:
        if self.model is None or not self.feature_names:
            return None
        coefs = self.model[-1].coef_
        return pd.Series(coefs, index=self.feature_names).sort_values(
            key=lambda s: s.abs(), ascending=False
        )


class ElasticNetModel(BaseEstimator):
    name = "elasticnet"

    def fit(self, X, y, feature_names=None):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        alpha = float(self.params.get("alpha", 0.001))
        l1 = float(self.params.get("l1_ratio", 0.5))
        self.model = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, random_state=self.seed),
        )
        self.model.fit(X, y)
        self.feature_names = list(feature_names or [])
        return self

    def feature_importance(self) -> pd.Series | None:
        if self.model is None or not self.feature_names:
            return None
        return pd.Series(self.model[-1].coef_, index=self.feature_names).sort_values(
            key=lambda s: s.abs(), ascending=False
        )


class RandomForestModel(BaseEstimator):
    name = "random_forest"

    def fit(self, X, y, feature_names=None):
        params = {
            "n_estimators": int(self.params.get("n_estimators", 80)),
            "max_depth": int(self.params.get("max_depth", 4)),
            "min_samples_leaf": int(self.params.get("min_samples_leaf", 500)),
            "max_features": self.params.get("max_features", 0.5),
            "n_jobs": int(self.params.get("n_jobs", 2)),
            "max_samples": float(self.params.get("max_samples", 0.4)),
            "random_state": self.seed,
        }
        self.model = RandomForestRegressor(**params)
        self.model.fit(X, y)
        self.feature_names = list(feature_names or [])
        return self

    def feature_importance(self) -> pd.Series | None:
        if self.model is None or not self.feature_names:
            return None
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(
            ascending=False
        )


class LightGBMModel(BaseEstimator):
    name = "lightgbm"

    def fit(self, X, y, feature_names=None):
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("lightgbm is not installed") from exc

        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "num_leaves": int(self.params.get("num_leaves", 31)),
            "min_data_in_leaf": int(self.params.get("min_data_in_leaf", 200)),
            "feature_fraction": float(self.params.get("feature_fraction", 0.7)),
            "bagging_fraction": float(self.params.get("bagging_fraction", 0.8)),
            "bagging_freq": int(self.params.get("bagging_freq", 1)),
            "lambda_l2": float(self.params.get("lambda_l2", 1.0)),
            "verbosity": -1,
            "num_threads": int(self.params.get("num_threads", 2)),
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
        }
        n_estimators = int(self.params.get("n_estimators", 250))
        train_set = lgb.Dataset(X, label=y, feature_name=list(feature_names or []))
        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[train_set],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        self.feature_names = list(feature_names or [])
        return self

    def feature_importance(self, importance_type: str = "gain") -> pd.Series | None:
        if self.model is None or not self.feature_names:
            return None
        values = self.model.feature_importance(importance_type=importance_type)
        return pd.Series(values, index=self.feature_names).sort_values(ascending=False)


REGISTRY: dict[str, type[BaseEstimator]] = {
    "ridge": RidgeModel,
    "elasticnet": ElasticNetModel,
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
}


def get_estimator(model_type: str, params: dict | None = None, seed: int = 42) -> BaseEstimator:
    key = (model_type or "ridge").lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model type {model_type!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[key](params=params, seed=seed)


__all__ = [
    "BaseEstimator",
    "ModelConfig",
    "RidgeModel",
    "ElasticNetModel",
    "RandomForestModel",
    "LightGBMModel",
    "get_estimator",
    "REGISTRY",
]
