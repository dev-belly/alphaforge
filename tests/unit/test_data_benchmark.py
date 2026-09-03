"""Regression tests for benchmark-series wiring in the data pipeline.

The benchmark must be attached to ``DataBundle`` as a *daily return* series
(not price levels) so the backtest engine, factor attribution and market
regime classification can consume it directly. Historically it was only
persisted and never returned, leaving every downstream relative-stat None.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.pipeline import DataPipeline


def test_bundle_benchmark_is_return_series():
    res = DataPipeline(provider="sample").run(
        start="2018-01-01", end="2024-12-31", index_id="SP500_SAMPLE", persist=False
    )
    bm = res.bundle.benchmark
    assert isinstance(bm, pd.Series), "benchmark must be attached as a Series"
    assert len(bm) > 200, "benchmark should span the full sample window"
    assert bm.notna().all(), "benchmark returns must have no NaNs"
    # A return series must sit near zero, not at the price level (hundreds).
    assert abs(bm.median()) < 0.05, "benchmark looks like price levels, not returns"
    assert np.isfinite(bm.to_numpy()).all()


def test_benchmark_drives_regime_classification():
    from alphaforge.risk.regime import classify_regime

    res = DataPipeline(provider="sample").run(
        start="2018-01-01", end="2024-12-31", index_id="SP500_SAMPLE", persist=False
    )
    labels = classify_regime(res.bundle.benchmark.dropna())
    # The full sample has enough history for all four regimes to appear.
    assert set(labels.dropna().unique()) == {
        "Bull/LowVol",
        "Bull/HighVol",
        "Bear/LowVol",
        "Bear/HighVol",
    }
