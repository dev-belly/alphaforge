"""A frozen dataset, identical forecast coverage and explicit cost scenarios."""

from __future__ import annotations

import copy
import gzip
import hashlib
import html
import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from alphaforge.backtest.engine import BacktestConfig, BacktestEngine
from alphaforge.backtest.metrics import performance_stats
from alphaforge.models.evaluation import evaluate_predictions
from alphaforge.models.pipeline import AlphaModelPipeline, signal_panel

MODEL_PARAMS = {
    "ridge": {"alpha": 1.0},
    "lightgbm": {
        "n_estimators": 100,
        "num_leaves": 15,
        "min_data_in_leaf": 200,
        "num_threads": 1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "learning_rate": 0.05,
        "lambda_l2": 1.0,
    },
}


def dataset_digest(dataset) -> str:
    digest = hashlib.sha256()
    for frame in (
        dataset.features,
        pd.DataFrame(
            {
                "date": dataset.dates,
                "symbol": dataset.symbols,
                "target": dataset.target,
                "forward_return": dataset.forward_returns,
            }
        ),
    ):
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        digest.update(json.dumps(list(frame.columns)).encode())
    return digest.hexdigest()


def validate_coverage(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    keys = ["date", "symbol", "fold", "realised"]
    if candidate.duplicated(["date", "symbol"]).any():
        raise ValueError("Comparison has overlapping/duplicate out-of-sample forecasts")
    if not np.isfinite(candidate["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("Comparison has nonfinite predictions")
    if not reference[keys].reset_index(drop=True).equals(candidate[keys].reset_index(drop=True)):
        raise ValueError("Methods do not share the same forecast dates, names, folds and outcomes")


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def ranked_weights(panel, scores: pd.DataFrame, top_n: int | None):
    """Equal weight top-N scores with symbol ties; no optimizer or size proxy."""

    def weights(decision, previous):
        available = scores.loc[decision].where(panel.universe.loc[decision]).dropna()
        if available.empty:
            return None
        ordered = sorted(available.index, key=lambda name: (-float(available[name]), str(name)))
        selected = ordered if top_n is None else ordered[:top_n]
        return pd.Series(1 / len(selected), index=selected, dtype=float)

    return weights


def run_comparison(
    state,
    destination: str | Path,
    *,
    seeds=(11, 29, 42),
    cost_multipliers=(1.0, 2.0),
    top_n=50,
    assurance=None,
):
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError("choose distinct uint32 seeds")
    if not cost_multipliers or any(not np.isfinite(x) or x <= 0 for x in cost_multipliers):
        raise ValueError("cost multipliers must be finite and positive")
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    provider = str(state.config.get("data", {}).get("provider", "sample"))
    synthetic = provider in {"sample", "synthetic", ""}
    if not synthetic and not assurance:
        raise ValueError("Real-data comparisons need an explicit source assurance record")
    dataset, panel = state.dataset, state.panel
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(state.config)
    reference = None
    cases, forecasts, folds = [], {}, []
    daily: list[dict[str, Any]] = []
    variants = [("ridge", "ridge", dataset), ("lightgbm", "lightgbm", dataset)]
    if "log_market_cap" in dataset.features.columns:
        reduced = replace(
            dataset,
            features=dataset.features.drop(columns="log_market_cap"),
            feature_names=[name for name in dataset.feature_names if name != "log_market_cap"],
        )
        variants.append(("ridge_without_size_feature", "ridge", reduced))
    for name, model_type, inputs in variants:
        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg.setdefault("project", {})["seed"] = seed
            cfg["model"]["seed"] = seed
            cfg["model"]["params"] = MODEL_PARAMS[model_type]
            wf = AlphaModelPipeline.from_config(inputs, cfg).run(model_type)
            predictions = wf.predictions
            if reference is None:
                reference = predictions
            validate_coverage(reference, predictions)
            forecasts[name, seed] = predictions
            for row in wf.fold_metrics.to_dict(orient="records"):
                folds.append({"method": name, "seed": seed, **row})
    assert reference is not None
    for name in ("equal_weight", "single_momentum"):
        predictions = reference.copy()
        if name == "equal_weight":
            predictions["prediction"] = 0.0
        else:
            if "mom_12_1" not in dataset.features.columns:
                raise ValueError("The fixed single-factor baseline requires mom_12_1")
            lookup = pd.DataFrame(
                {
                    "date": dataset.dates,
                    "symbol": dataset.symbols,
                    "momentum": dataset.features["mom_12_1"].to_numpy(),
                }
            )
            predictions = predictions.drop(columns="prediction").merge(
                lookup, on=["date", "symbol"], how="left", validate="one_to_one"
            )
            predictions = predictions.rename(columns={"momentum": "prediction"})
        validate_coverage(reference, predictions)
        forecasts[name, None] = predictions
    shared_index = None
    for (name, seed), predictions in forecasts.items():
        scores = signal_panel(predictions, panel.dates, panel.symbols).ffill()
        evaluation = evaluate_predictions(
            predictions,
            model_name=name,
            horizon=int(dataset.metadata["horizon"]),
            fold_id=predictions["fold"],
        )
        for multiplier in cost_multipliers:
            cost = dict(config.get("cost", {}))
            for key in ("commission_bps", "slippage_bps", "impact_coeff_bps"):
                cost[key] = (
                    float(
                        cost.get(
                            key,
                            {"commission_bps": 2, "slippage_bps": 5, "impact_coeff_bps": 10}[key],
                        )
                    )
                    * multiplier
                )
            bt = BacktestEngine(
                panel,
                weight_fn=ranked_weights(panel, scores, None if name == "equal_weight" else top_n),
                cost_model=cost,
                config=BacktestConfig.from_dict(config["backtest"]),
            ).run()
            if bt.diagnostics["n_skipped_rebalances"]:
                raise ValueError(
                    f"{name} has failed rebalances; comparison cannot silently drop them"
                )
            if shared_index is None:
                shared_index = bt.returns.index
            if not bt.returns.index.equals(shared_index):
                raise ValueError("Comparison backtest calendars differ")
            cases.append(
                {
                    "method": name,
                    "seed": seed,
                    "cost_multiplier": multiplier,
                    "features": len(dataset.feature_names)
                    - int(name == "ridge_without_size_feature"),
                    "rank_ic_mean": evaluation.summary.get("rank_ic_mean"),
                    "icir": evaluation.summary.get("icir"),
                    **bt.summary(),
                    "n_rebalances": bt.diagnostics["n_rebalances"],
                    "backtest_start": str(bt.returns.index.min().date()),
                    "backtest_end": str(bt.returns.index.max().date()),
                }
            )
            assert bt.gross_equity is not None
            gross = bt.gross_equity.pct_change().fillna(0.0)
            daily.extend(
                {
                    "method": name,
                    "seed": seed,
                    "cost_multiplier": multiplier,
                    "date": str(day.date()),
                    "net_return": float(bt.returns.loc[day]),
                    "gross_return": float(gross.loc[day]),
                }
                for day in bt.returns.index
            )
    frame = pd.DataFrame(cases)
    aggregates = []
    for (name, multiplier), group in frame.groupby(["method", "cost_multiplier"], sort=True):
        aggregates.append(
            {
                "method": name,
                "cost_multiplier": multiplier,
                "runs": len(group),
                **{
                    f"{metric}_{stat}": (
                        float(getattr(group[metric], stat)(ddof=0))
                        if stat == "std"
                        else float(group[metric].mean())
                    )
                    for metric in ("cagr", "sharpe", "max_drawdown", "rank_ic_mean")
                    for stat in ("mean", "std")
                },
            }
        )
    summary = _clean(
        {
            "data_kind": "synthetic" if synthetic else "exploratory_real_market",
            "source_assurance": assurance,
            "dataset_sha256": dataset_digest(dataset),
            "dataset": {
                "samples": len(dataset),
                "features": dataset.feature_names,
                "oos_forecasts": len(reference),
                "horizon_sessions": dataset.metadata["horizon"],
                "forecast_start": str(reference["date"].min().date()),
                "forecast_end": str(reference["date"].max().date()),
            },
            "protocol": {
                "seeds": list(seeds),
                "model_params": MODEL_PARAMS,
                "walk_forward": config["model"]["walk_forward"],
                "backtest": config["backtest"],
                "cost": config["cost"],
                "cost_multipliers": list(cost_multipliers),
                "portfolio": f"long-only equal-weight top {top_n}; equal-weight baseline holds all forecastable eligible names",
                "ties": "score descending then symbol ascending",
                "signal_availability": "forecast panel forward-fill only; no backfill",
                "ablation": "drop explicit log_market_cap predictor only; earlier size neutralization remains",
                "model_selection": "none; all seeds/cost cells reported; failed cells raise",
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "limitations": [
                "Seed variation is not market sampling uncertainty or proof of tradable alpha.",
                "Cost inputs are fixed scenarios, not measured execution costs.",
                "The equal-weight baseline differs in diversification; its holdings rule is explicit.",
                "Source hashes establish identity, not complete historical data quality.",
            ],
            "aggregates": aggregates,
            "cases": cases,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    frame.to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(folds).to_csv(output / "folds.csv", index=False, lineterminator="\n")
    pd.DataFrame(daily).to_csv(output / "daily_returns.csv", index=False, lineterminator="\n")
    (output / "daily_returns.csv.gz").write_bytes(
        gzip.compress((output / "daily_returns.csv").read_bytes(), mtime=0)
    )
    table = "".join(
        f"<tr><td>{html.escape(row['method'])}</td><td>{row['cost_multiplier']:g}×</td>"
        f"<td>{row['runs']}</td><td>{row['cagr_mean']:.2%}</td>"
        f"<td>{row['sharpe_mean']:.3f} ± {row['sharpe_std']:.3f}</td>"
        f"<td>{row['max_drawdown_mean']:.2%}</td></tr>"
        for row in summary["aggregates"]
    )
    notes = list(summary["limitations"]) + (assurance or {}).get("limitations", [])
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AlphaForge · Frozen comparison</title>
<style>:root{{font-family:system-ui;color-scheme:dark;background:#111319;color:#eee}}main{{max-width:1100px;margin:auto;padding:2rem}}h1{{font-size:3rem}}a{{color:#d7fa73}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.6rem;text-align:left;border-bottom:1px solid #444;white-space:nowrap}}.scroll{{overflow:auto}}aside{{padding:1rem;border-left:3px solid #ffb994}}</style>
<main><p>{summary["data_kind"].upper()}</p><h1>One dataset.<br>Declared comparisons.</h1>
<p>{len(dataset):,} samples · {len(dataset.feature_names)} usable features · {len(reference):,} shared forecasts.
Models, the fixed momentum baseline and equal-weight baseline share dates, outcomes, execution lag and cost rules.</p>
<aside>{"<br>".join(html.escape(note) for note in notes)}</aside>
<h2>All methods and cost scenarios</h2><div class="scroll"><table><tr><th>Method</th><th>Costs</th><th>Runs</th><th>Net CAGR</th><th>Net Sharpe mean ± seed std</th><th>Max drawdown</th></tr>{table}</table></div>
<p><a href="summary.json">Source identity, exact protocol and all runs</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="folds.csv">Fold results</a> · <a href="daily_returns.csv.gz">Daily returns (gzip CSV)</a> · <a href="manifest.json">Hashes</a></p>
<p>This is a protocol comparison. Its real-market variant remains exploratory until the declared source limitations are resolved. No test result selects a model or seed for deployment.</p></main></html>"""
    (output / "index.html").write_text(report, encoding="utf-8")
    names = ("summary.json", "metrics.csv", "folds.csv", "daily_returns.csv.gz", "index.html")
    manifest = {
        "dataset_sha256": summary["dataset_sha256"],
        "files_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def verify_comparison(directory: str | Path):
    """Recompute portfolio headlines from exported returns, without fitting models."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    names = {"summary.json", "metrics.csv", "folds.csv", "daily_returns.csv.gz", "index.html"}
    if set(manifest["files_sha256"]) != names:
        raise ValueError("Unexpected comparison artifact inventory")
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Comparison artifact digest mismatch: {name}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(directory / "daily_returns.csv.gz")
    daily["date"] = pd.to_datetime(daily["date"])
    if not np.isfinite(daily[["net_return", "gross_return"]].to_numpy()).all():
        raise ValueError("Nonfinite comparison returns")
    grouped = daily.groupby(["method", "seed", "cost_multiplier"], dropna=False, sort=False)
    if len(grouped) != len(summary["cases"]):
        raise ValueError("Comparison case coverage mismatch")
    shared_dates = None
    verified_cases = []
    metrics = pd.read_csv(directory / "metrics.csv")
    for case in summary["cases"]:
        rows = daily[
            (daily["method"] == case["method"])
            & (daily["cost_multiplier"] == case["cost_multiplier"])
        ]
        rows = (
            rows[rows["seed"].isna()]
            if case["seed"] is None
            else rows[rows["seed"] == case["seed"]]
        )
        if rows.empty or rows["date"].duplicated().any():
            raise ValueError("Missing or duplicate comparison return dates")
        dates = pd.DatetimeIndex(rows["date"])
        if shared_dates is None:
            shared_dates = dates
        if not dates.equals(shared_dates):
            raise ValueError("Comparison return calendars disagree")
        actual = performance_stats(pd.Series(rows["net_return"].to_numpy(), index=dates))
        gross = performance_stats(pd.Series(rows["gross_return"].to_numpy(), index=dates))
        stated_csv = metrics[
            (metrics["method"] == case["method"])
            & (metrics["cost_multiplier"] == case["cost_multiplier"])
        ]
        stated_csv = (
            stated_csv[stated_csv["seed"].isna()]
            if case["seed"] is None
            else stated_csv[stated_csv["seed"] == case["seed"]]
        )
        if len(stated_csv) != 1:
            raise ValueError("Comparison metrics CSV coverage mismatch")
        for key in ("cagr", "sharpe", "max_drawdown", "ann_vol", "total_return"):
            for title, computed in ((key, actual[key]), ("gross_" + key, gross[key])):
                if title not in case:
                    continue
                if (
                    not math.isfinite(float(case[title]))
                    or abs(computed - case[title]) > 1e-8
                    or abs(computed - float(stated_csv.iloc[0][title])) > 1e-8
                ):
                    raise ValueError(f"Comparison {title} disagrees with exported returns")
        verified_cases.append(
            {
                **case,
                "cagr": actual["cagr"],
                "sharpe": actual["sharpe"],
                "max_drawdown": actual["max_drawdown"],
            }
        )
    verified = pd.DataFrame(verified_cases)
    for stated in summary["aggregates"]:
        group = verified[
            (verified["method"] == stated["method"])
            & (verified["cost_multiplier"] == stated["cost_multiplier"])
        ]
        if len(group) != stated["runs"]:
            raise ValueError("Comparison aggregate run count mismatch")
        for key in ("cagr", "sharpe", "max_drawdown"):
            if (
                abs(float(group[key].mean()) - stated[key + "_mean"]) > 1e-8
                or abs(float(group[key].std(ddof=0)) - stated[key + "_std"]) > 1e-8
            ):
                raise ValueError("Comparison aggregates disagree with exported returns")
    assert shared_dates is not None
    return {
        "status": "verified",
        "cases": len(verified_cases),
        "return_dates_per_case": len(shared_dates),
        "dataset_sha256": manifest["dataset_sha256"],
    }
