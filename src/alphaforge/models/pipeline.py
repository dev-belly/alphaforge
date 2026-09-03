"""Walk-forward training of cross-sectional alpha models.

The engine trains one model per fold on strictly earlier data, predicts the
next out-of-sample block, and concatenates the predictions.  Hyper-parameters
are never tuned on the concatenated out-of-sample predictions - that would be
the same leakage in a slower costume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.models.dataset import AlphaDataset
from alphaforge.models.estimators import ModelConfig, get_estimator
from alphaforge.models.evaluation import ModelEvaluation, evaluate_predictions
from alphaforge.models.split import WalkForwardConfig, WalkForwardSplitter
from alphaforge.utils.logging import Timer, get_logger

log = get_logger("models.pipeline")


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # date, symbol, prediction, realised, fold
    models: list = field(default_factory=list)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluation: ModelEvaluation | None = None
    config: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return self.evaluation.summary if self.evaluation else {}


class AlphaModelPipeline:
    """Fits and validates one model type under walk-forward CV."""

    def __init__(
        self,
        dataset: AlphaDataset,
        model_config: ModelConfig | None = None,
        split_config: WalkForwardConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model_config = model_config or ModelConfig()
        self.split_config = split_config or WalkForwardConfig(
            purge_days=int(dataset.metadata.get("horizon", 21)),
            embargo_days=int(dataset.metadata.get("horizon", 21)),
        )

    @classmethod
    def from_config(cls, dataset: AlphaDataset, cfg: dict) -> AlphaModelPipeline:
        mcfg = ModelConfig.from_dict(cfg.get("model", {}))
        wcfg_dict = dict((cfg.get("model", {}) or {}).get("walk_forward", {}) or {})
        horizon = int(dataset.metadata.get("horizon", 21))
        wcfg = WalkForwardConfig(
            train_years=int(wcfg_dict.get("train_years", 4)),
            test_years=int(wcfg_dict.get("test_years", 1)),
            step_years=int(wcfg_dict.get("step_years", 1)),
            purge_days=int(wcfg_dict.get("purge_days", horizon)),
            embargo_days=int(wcfg_dict.get("embargo_days", horizon)),
            expanding=bool(wcfg_dict.get("expanding", True)),
        )
        return cls(dataset=dataset, model_config=mcfg, split_config=wcfg)

    # ------------------------------------------------------------------
    def run(self, model_type: str | None = None) -> WalkForwardResult:
        mtype = model_type or self.model_config.type
        ds = self.dataset
        dates = ds.date_index()
        folds = WalkForwardSplitter(self.split_config).split(dates)
        if not folds:
            raise RuntimeError("Walk-forward splitter produced no folds - extend the sample")

        X_all = ds.features.to_numpy(dtype=float)
        y_all = ds.target.to_numpy(dtype=float)
        date_all = pd.DatetimeIndex(ds.dates)
        sym_all = ds.symbols.to_numpy()
        fwd_all = ds.forward_returns.to_numpy(dtype=float)

        # Map each sample row to its position in the sorted unique date index.
        date_pos = pd.Series(np.arange(len(dates)), index=dates)
        row_pos = date_pos.reindex(date_all).to_numpy()

        chunks = []
        models = []
        importances = []
        fold_ids = []

        with Timer(f"models.walk_forward[{mtype}]", log):
            for fold in folds:
                train_mask_rows = fold.train_mask[row_pos]
                test_mask_rows = fold.test_mask[row_pos]
                if train_mask_rows.sum() < 100 or test_mask_rows.sum() < 10:
                    log.warning(f"Skipping {fold} - insufficient samples")
                    continue

                est = get_estimator(mtype, self.model_config.params, seed=self.model_config.seed)
                est.fit(
                    X_all[train_mask_rows],
                    y_all[train_mask_rows],
                    feature_names=list(ds.feature_names),
                )
                preds = est.predict(X_all[test_mask_rows])
                models.append(est)

                chunk = pd.DataFrame(
                    {
                        "date": date_all[test_mask_rows],
                        "symbol": sym_all[test_mask_rows],
                        "prediction": preds,
                        "realised": fwd_all[test_mask_rows],
                        "fold": fold.index,
                    }
                )
                chunks.append(chunk)
                fold_ids.append(fold.index)

                imp = est.feature_importance()
                if imp is not None:
                    s = imp.rename(f"fold_{fold.index}")
                    importances.append(s)
                log.info(f"{mtype} fold {fold.index}: {len(chunk):,} predictions")

        if not chunks:
            raise RuntimeError("Walk-forward produced no predictions")

        predictions = pd.concat(chunks, ignore_index=True).sort_values(["date", "symbol"])
        predictions = predictions.reset_index(drop=True)

        importance = pd.DataFrame()
        if importances:
            importance = pd.concat(importances, axis=1)
            importance["mean"] = importance.mean(axis=1)
            importance["std"] = importance.iloc[:, :-1].std(axis=1)
            importance = importance.sort_values("mean", ascending=False)

        evaluation = evaluate_predictions(
            predictions,
            model_name=mtype,
            horizon=int(ds.metadata.get("horizon", 21)),
            fold_id=predictions["fold"],
        )
        evaluation.feature_importance = importance

        return WalkForwardResult(
            predictions=predictions,
            models=models,
            fold_metrics=evaluation.fold_metrics,
            importance=importance,
            evaluation=evaluation,
            config={
                "model_type": mtype,
                "params": self.model_config.params,
                "split": self.split_config.__dict__,
                "horizon": ds.metadata.get("horizon", 21),
            },
        )


def run_model_comparison(
    dataset: AlphaDataset,
    model_types: list[str] | None = None,
    cfg: dict | None = None,
) -> dict[str, WalkForwardResult]:
    """Fit several model types under the *identical* walk-forward protocol."""
    cfg = cfg or {}
    types = model_types or ["ridge", "elasticnet", "random_forest", "lightgbm"]
    results: dict[str, WalkForwardResult] = {}
    for mtype in types:
        params = dict((cfg.get("model", {}) or {}).get("params", {}) or {})
        if mtype in (cfg.get("model_params", {}) or {}):
            params = dict(cfg["model_params"][mtype])
        try:
            pipe = AlphaModelPipeline.from_config(dataset, cfg)
            pipe.model_config.type = mtype
            pipe.model_config.params = params
            results[mtype] = pipe.run(model_type=mtype)
        except Exception as exc:  # noqa: BLE001 - keep the comparison grid running
            log.error(f"Model {mtype} failed: {exc}")
    return results


def signal_panel(
    predictions: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index
) -> pd.DataFrame:
    """Pivot out-of-sample predictions back into a (dates x symbols) score panel."""
    out = predictions.pivot_table(
        index="date", columns="symbol", values="prediction", aggfunc="last"
    )
    return out.reindex(index=dates, columns=symbols)


__all__ = ["AlphaModelPipeline", "WalkForwardResult", "run_model_comparison", "signal_panel"]
