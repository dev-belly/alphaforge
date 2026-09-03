"""Quant-appropriate model evaluation.

Accuracy is meaningless for a cross-sectional return model.  What matters is
whether the predicted ranking translates into a tradeable spread after costs,
so every metric below is expressed in ranking / portfolio terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("models.evaluation")


def rank_ic(pred: pd.Series, realised: pd.Series) -> float:
    """Spearman rank correlation between prediction and realised return."""
    df = pd.concat([pred, realised], axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank()))


def _pivot_predictions(predictions: pd.DataFrame, value: str) -> pd.DataFrame:
    return predictions.pivot_table(index="date", columns="symbol", values=value, aggfunc="last")


def daily_rank_ic(predictions: pd.DataFrame, min_names: int = 5) -> pd.Series:
    """Per-date rank IC from a long frame with date / pred / realised columns.

    Vectorised by pivoting to the (dates x symbols) grid and reusing the same
    row-wise correlation kernel as the factor engine.
    """
    pred = _pivot_predictions(predictions, "prediction")
    real = _pivot_predictions(predictions, "realised").reindex(
        index=pred.index, columns=pred.columns
    )
    common = pred.notna() & real.notna()
    ok = common.sum(axis=1) >= min_names
    p_rank = pred.where(ok).rank(axis=1)
    r_rank = real.where(ok).rank(axis=1)
    from alphaforge.factors.evaluation import _rowwise_corr

    return _rowwise_corr(p_rank, r_rank, common & ok.values[:, None]).rename("rank_ic")


def icir(ic: pd.Series) -> float:
    ic = ic.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return float("nan")
    return float(ic.mean() / ic.std())


def quantile_returns(predictions: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Mean realised return per prediction quantile, per date."""
    df = predictions[["date", "prediction", "realised"]].dropna()
    if df.empty:
        return pd.DataFrame()
    ranks = df.groupby("date")["prediction"].rank(method="first")
    sizes = df.groupby("date")["prediction"].transform("size")
    bucket = np.floor((ranks / sizes.replace(0, np.nan)) * n_quantiles).clip(upper=n_quantiles - 1)

    grouped = df.groupby("date")
    out = pd.DataFrame(index=sorted(df["date"].unique()))
    # ``valid`` must be indexed by date, otherwise ``.where`` misaligns and
    # silently produces an all-NaN frame.
    valid = grouped.size() >= n_quantiles * 2
    for qi in range(n_quantiles):
        mask = bucket == qi
        numerator = df["realised"].where(mask).groupby(df["date"]).sum()
        denominator = mask.groupby(df["date"]).sum().replace(0, np.nan)
        out[f"q{qi + 1}"] = (numerator / denominator).where(valid)
    out.index.name = "date"
    return out


def top_quantile_stats(predictions: pd.DataFrame, n_quantiles: int = 5) -> dict:
    q = quantile_returns(predictions, n_quantiles=n_quantiles)
    if q.empty:
        return {}
    spread = q[f"q{n_quantiles}"] - q["q1"]
    return {
        "top_quantile_return": float(q[f"q{n_quantiles}"].mean()),
        "bottom_quantile_return": float(q["q1"].mean()),
        "long_short_spread": float(spread.mean()),
        "long_short_ir": float(spread.mean() / spread.std()) if spread.std() else float("nan"),
        "hit_ratio": float((spread > 0).mean()),
    }


def prediction_turnover(predictions: pd.DataFrame, top_frac: float = 0.2) -> float:
    """Average one-way turnover of the top-``top_frac`` portfolio."""
    frames = []
    for date, g in predictions.groupby("date", sort=True):
        if len(g) < 5:
            continue
        n_top = max(int(len(g) * top_frac), 1)
        top = set(g.nlargest(n_top, "prediction")["symbol"])
        frames.append((date, top))
    if not frames:
        return float("nan")
    turnovers = []
    prev: set | None = None
    for _, top in frames:
        if prev is not None and top:
            turnovers.append(len(top - prev) / max(len(top), 1))
        prev = top
    return float(np.mean(turnovers)) if turnovers else float("nan")


@dataclass
class ModelEvaluation:
    """Full out-of-sample evaluation record for one model run."""

    model_name: str
    predictions: pd.DataFrame  # date, symbol, prediction, realised
    ic_series: pd.Series
    fold_metrics: pd.DataFrame
    yearly_metrics: pd.DataFrame
    quantile_returns: pd.DataFrame
    summary: dict = field(default_factory=dict)
    feature_importance: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        return {"model": self.model_name, **self.summary}


def evaluate_predictions(
    predictions: pd.DataFrame,
    model_name: str = "model",
    horizon: int = 21,
    fold_id: pd.Series | None = None,
    periods_per_year: float = 252.0,
) -> ModelEvaluation:
    """Build the complete evaluation from a long prediction frame."""
    ic = daily_rank_ic(predictions)
    ic_clean = ic.dropna()
    eff_n = max(len(ic_clean) / max(horizon, 1), 2.0)
    t_stat = (
        float(ic_clean.mean() / ic_clean.std() * np.sqrt(eff_n)) if ic_clean.std() else float("nan")
    )

    q = quantile_returns(predictions)
    tq = top_quantile_stats(predictions)
    turn = prediction_turnover(predictions)

    yearly = pd.DataFrame({"rank_ic": ic}).dropna()
    yearly = yearly.groupby(yearly.index.year).agg(["mean", "std", "count"])
    yearly.columns = ["_".join(c) for c in yearly.columns]
    yearly["icir"] = yearly["rank_ic_mean"] / yearly["rank_ic_std"].replace(0, np.nan)
    yearly = yearly.reset_index().rename(columns={"index": "year", "date": "year"})

    folds = pd.DataFrame()
    if fold_id is not None:
        tmp = pd.DataFrame({"fold": fold_id.to_numpy(), "date": predictions["date"].to_numpy()})
        tmp = tmp.merge(ic.rename("rank_ic").reset_index(), on="date", how="left")
        folds = (
            tmp.groupby("fold")
            .agg(
                rank_ic_mean=("rank_ic", "mean"),
                rank_ic_std=("rank_ic", "std"),
                n_days=("date", "nunique"),
                n_obs=("rank_ic", "count"),
            )
            .reset_index()
        )
        folds["icir"] = folds["rank_ic_mean"] / folds["rank_ic_std"].replace(0, np.nan)
        folds = folds.rename(columns={"n_obs": "n_days"})

    ann_factor = periods_per_year / max(horizon, 1)
    summary = {
        "model": model_name,
        "rank_ic_mean": float(ic_clean.mean()) if len(ic_clean) else float("nan"),
        "rank_ic_std": float(ic_clean.std()) if len(ic_clean) else float("nan"),
        "icir": icir(ic),
        "t_stat": t_stat,
        "positive_ic_ratio": float((ic_clean > 0).mean()) if len(ic_clean) else float("nan"),
        "n_periods": int(len(ic_clean)),
        "turnover": float(turn),
        "ann_long_short": float(tq.get("long_short_spread", np.nan)) * ann_factor,
        "long_short_ir": tq.get("long_short_ir", float("nan")),
        "hit_ratio": tq.get("hit_ratio", float("nan")),
        **tq,
    }
    log.info(
        f"{model_name}: RankIC={summary['rank_ic_mean']:+.4f}  ICIR={summary['icir']:+.3f}  "
        f"t={t_stat:+.2f}  turnover={summary['turnover']:.2f}"
    )
    return ModelEvaluation(
        model_name=model_name,
        predictions=predictions,
        ic_series=ic,
        fold_metrics=folds,
        yearly_metrics=yearly,
        quantile_returns=q,
        summary=summary,
    )


def compare_models(evaluations: dict[str, ModelEvaluation]) -> pd.DataFrame:
    rows = [e.to_dict() for e in evaluations.values()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = [
        "model",
        "rank_ic_mean",
        "rank_icir" if "rank_icir" in df else "icir",
        "t_stat",
        "turnover",
        "ann_long_short",
        "long_short_ir",
        "hit_ratio",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values("rank_ic_mean", ascending=False).reset_index(drop=True)


__all__ = [
    "ModelEvaluation",
    "rank_ic",
    "daily_rank_ic",
    "icir",
    "quantile_returns",
    "top_quantile_stats",
    "prediction_turnover",
    "evaluate_predictions",
    "compare_models",
]
