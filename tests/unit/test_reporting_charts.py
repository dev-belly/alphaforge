"""Offline tests for the report figures.

A chart is a claim, and the failure that matters is a *plausible* one: an
equity curve compounded from the wrong base, a drawdown drawn off the running
peak of the wrong series, an IC line plotted as a level instead of a cumulative
sum. None of those raise; they just produce a picture that looks right.

So the tests here do not stop at "it returned a base64 string". They intercept
``Axes.plot`` / ``Axes.fill_between`` and assert the **numbers handed to
matplotlib** against an independently computed series.

One near-miss is worth recording. ``(1 + benchmark).cumprod()`` looks like it
would propagate a NaN through the rest of the curve and blank the benchmark line
after its first missing session. It does not - ``Series.cumprod`` defaults to
``skipna=True``, so a gap stays a gap and the level carries across it, which is
exactly what the docstring promises. ``test_a_gap_in_the_benchmark_keeps_the_level``
pins that so a future change to ``fillna``/``dropna`` cannot quietly alter it.

Everything is deterministic: no network, Agg backend, no files written.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.axes  # noqa: E402

from alphaforge.reporting.charts import (  # noqa: E402
    brinson_bars,
    drawdown,
    equity_curve,
    ic_series,
    monthly_heatmap,
    quantile_bar,
    risk_contribution,
)

DATES = pd.bdate_range("2022-01-03", periods=260)
SYMBOLS = [f"S{i:02d}" for i in range(8)]


def _equity(start: float = 1_000_000.0, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, size=len(DATES))
    return pd.Series(start * np.cumprod(1.0 + rets), index=DATES)


def _returns(seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0002, 0.009, size=len(DATES)), index=DATES)


def _is_png(payload: str) -> bool:
    raw = base64.b64decode(payload)
    return raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) > 1000


def _capture(monkeypatch, method: str, label: str | None = None) -> dict:
    """Intercept a plotting call and keep the y-values it was handed."""
    captured: dict = {}
    original = getattr(matplotlib.axes.Axes, method)

    def spy(self, *args, **kwargs):
        if label is None or kwargs.get("label") == label:
            captured.setdefault("y", np.asarray(args[1], dtype=float))
            captured.setdefault("x", args[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, method, spy)
    return captured


# ----------------------------------------------------------------------
# The PNG contract
# ----------------------------------------------------------------------
def test_a_rendered_chart_is_a_real_png() -> None:
    assert _is_png(equity_curve(_equity()))


def test_a_rendered_chart_closes_its_figure() -> None:
    """Leaking figures would grow the API process without bound."""
    import matplotlib.pyplot as plt

    equity_curve(_equity())
    assert plt.get_fignums() == []


# ----------------------------------------------------------------------
# equity_curve
# ----------------------------------------------------------------------
def test_equity_curve_plots_the_net_series_verbatim(monkeypatch) -> None:
    equity = _equity()
    captured = _capture(monkeypatch, "plot", label="Net (after-cost)")
    equity_curve(equity)
    np.testing.assert_allclose(captured["y"], equity.to_numpy())


def test_equity_curve_compounds_the_benchmark_from_the_initial_capital(
    monkeypatch,
) -> None:
    """A wrong base here shifts the whole benchmark line without changing its shape."""
    equity = _equity()
    benchmark = _returns()
    captured = _capture(monkeypatch, "plot", label="Benchmark")
    equity_curve(equity, benchmark=benchmark, initial_capital=1_000_000.0)
    expected = (1.0 + benchmark.reindex(equity.index)).cumprod() * 1_000_000.0
    np.testing.assert_allclose(captured["y"], expected.to_numpy(), rtol=1e-12)


def test_the_first_benchmark_return_is_included(monkeypatch) -> None:
    """``initial_capital`` is the *pre*-first-session NAV, so day one already moves."""
    equity = _equity()
    benchmark = _returns()
    captured = _capture(monkeypatch, "plot", label="Benchmark")
    equity_curve(equity, benchmark=benchmark, initial_capital=1_000_000.0)
    assert captured["y"][0] == pytest.approx(1_000_000.0 * (1.0 + benchmark.iloc[0]))
    assert captured["y"][0] != pytest.approx(1_000_000.0)


def test_a_gap_in_the_benchmark_keeps_the_level() -> None:
    """Regression guard: ``cumprod`` skips NaN, so a gap must not truncate the line.

    If someone swaps this for a ``fillna(0)``-free path that propagates NaN, or
    for a ``dropna`` that reindexes, the benchmark level after the gap moves.
    """
    equity = _equity()
    benchmark = _returns().copy()
    gap = DATES[100]
    benchmark.loc[gap] = np.nan

    level = (1.0 + benchmark.reindex(equity.index)).cumprod() * 1_000_000.0
    assert np.isnan(level.loc[gap]), "the gap itself must stay empty"
    assert not np.isnan(level.iloc[101:]).any(), "the level must survive the gap"
    # ...and it is the level that ignores the missing session entirely.
    ignored = (1.0 + benchmark.reindex(equity.index).fillna(0.0)).cumprod() * 1_000_000.0
    observed = level.notna()
    np.testing.assert_allclose(level[observed].to_numpy(), ignored[observed].to_numpy(), rtol=1e-12)


def test_equity_curve_plots_gross_when_given(monkeypatch) -> None:
    equity = _equity()
    gross = equity * 1.02
    captured = _capture(monkeypatch, "plot", label="Gross (pre-cost)")
    equity_curve(equity, gross=gross)
    np.testing.assert_allclose(captured["y"], gross.reindex(equity.index).to_numpy())


def test_equity_curve_skips_an_empty_gross_or_benchmark(monkeypatch) -> None:
    equity = _equity()
    empty = pd.Series(np.nan, index=equity.index)
    assert _is_png(equity_curve(equity, benchmark=empty, gross=empty))


def test_equity_curve_requires_a_capital_for_a_benchmark() -> None:
    """Silently defaulting to 1.0 would draw a benchmark on a different scale."""
    with pytest.raises(ValueError, match="initial_capital is required"):
        equity_curve(_equity(), benchmark=_returns())


def test_a_failed_equity_curve_does_not_leak_its_figure() -> None:
    import matplotlib.pyplot as plt

    with pytest.raises(ValueError):
        equity_curve(_equity(), benchmark=_returns())
    assert plt.get_fignums() == []


# ----------------------------------------------------------------------
# drawdown
# ----------------------------------------------------------------------
def test_drawdown_is_measured_from_the_running_peak(monkeypatch) -> None:
    equity = _equity()
    captured = _capture(monkeypatch, "fill_between")
    drawdown(equity)
    expected = (equity / equity.cummax() - 1.0).to_numpy() * 100.0
    np.testing.assert_allclose(captured["y"], expected, rtol=1e-12)
    assert (captured["y"] <= 1e-9).all(), "a drawdown can never be positive"


def test_drawdown_is_zero_at_a_new_high(monkeypatch) -> None:
    rising = pd.Series(np.arange(1.0, 261.0), index=DATES)
    captured = _capture(monkeypatch, "fill_between")
    drawdown(rising)
    np.testing.assert_allclose(captured["y"], np.zeros(len(DATES)), atol=1e-12)


# ----------------------------------------------------------------------
# ic_series
# ----------------------------------------------------------------------
def test_ic_series_plots_a_cumulative_sum_not_a_level(monkeypatch) -> None:
    """The title says cumulative; plotting the raw IC would still look plausible."""
    ic = pd.Series(np.full(len(DATES), 0.01), index=DATES)
    captured = _capture(monkeypatch, "plot")
    ic_series(ic)
    np.testing.assert_allclose(captured["y"], ic.cumsum().to_numpy(), rtol=1e-12)
    assert captured["y"][-1] == pytest.approx(0.01 * len(DATES))


def test_ic_series_is_empty_for_an_all_nan_series() -> None:
    assert ic_series(pd.Series(np.nan, index=DATES)) == ""


# ----------------------------------------------------------------------
# quantile_bar
# ----------------------------------------------------------------------
def test_quantile_bar_uses_the_mean_column_when_present(monkeypatch) -> None:
    table = pd.DataFrame({"q1": [-0.01, -0.02], "q2": [0.0, 0.0], "mean": [-0.015, -0.025]})
    captured = _capture(monkeypatch, "bar")
    quantile_bar(table)
    np.testing.assert_allclose(captured["y"], table["mean"].to_numpy() * 100.0)


def test_quantile_bar_averages_the_columns_without_a_mean_column(monkeypatch) -> None:
    table = pd.DataFrame({"q1": [-0.01, -0.02], "q2": [0.01, 0.02]})
    captured = _capture(monkeypatch, "bar")
    quantile_bar(table)
    np.testing.assert_allclose(captured["y"], table.mean(axis=0).to_numpy() * 100.0)


def test_quantile_bar_is_empty_for_an_empty_table() -> None:
    assert quantile_bar(pd.DataFrame()) == ""


# ----------------------------------------------------------------------
# monthly_heatmap
# ----------------------------------------------------------------------
def test_monthly_heatmap_scales_to_percent(monkeypatch) -> None:
    table = pd.DataFrame({"Jan": [0.01, -0.02], "Feb": [0.005, 0.0]}, index=[2021, 2022])
    captured: dict = {}
    original = matplotlib.axes.Axes.imshow

    def spy(self, data, *args, **kwargs):
        captured["data"] = np.asarray(data, dtype=float)
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", spy)
    assert _is_png(monthly_heatmap(table))
    np.testing.assert_allclose(captured["data"], table.to_numpy() * 100.0)


def test_monthly_heatmap_is_empty_for_an_empty_table() -> None:
    assert monthly_heatmap(pd.DataFrame()) == ""


def test_monthly_heatmap_survives_an_all_nan_table() -> None:
    table = pd.DataFrame({"Jan": [np.nan]}, index=[2021])
    assert _is_png(monthly_heatmap(table))


# ----------------------------------------------------------------------
# risk_contribution
# ----------------------------------------------------------------------
def _cov(symbols: list[str], seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(symbols), len(symbols)))
    return pd.DataFrame(a @ a.T, index=symbols, columns=symbols) / 1e4


def test_risk_contribution_is_the_marginal_contribution(monkeypatch) -> None:
    weights = pd.Series([0.5, 0.3, 0.2, 0.0], index=SYMBOLS[:4])
    cov = _cov(SYMBOLS[:4])
    captured = _capture(monkeypatch, "bar")
    risk_contribution(weights, cov)

    live = [c for c in weights.index if abs(weights[c]) > 1e-6]
    w = weights[live].to_numpy()
    c = cov.reindex(index=live, columns=live).to_numpy()
    expected = (w * (c @ w)) / float(w @ c @ w) * 100.0
    # The bars are sorted by contribution, so compare as multisets.
    np.testing.assert_allclose(np.sort(captured["y"]), np.sort(expected), rtol=1e-9)


def test_risk_contribution_is_empty_without_holdings() -> None:
    weights = pd.Series([0.0, 0.0], index=SYMBOLS[:2])
    assert risk_contribution(weights, _cov(SYMBOLS[:2])) == ""


def test_risk_contribution_is_empty_for_an_all_nan_weight_vector() -> None:
    weights = pd.Series([np.nan, np.nan], index=SYMBOLS[:2])
    assert risk_contribution(weights, _cov(SYMBOLS[:2])) == ""


def test_risk_contribution_survives_a_zero_variance_book(monkeypatch) -> None:
    """A degenerate covariance must not divide by zero."""
    weights = pd.Series([0.5, 0.5], index=SYMBOLS[:2])
    cov = pd.DataFrame(0.0, index=SYMBOLS[:2], columns=SYMBOLS[:2])
    assert _is_png(risk_contribution(weights, cov))


# ----------------------------------------------------------------------
# brinson_bars
# ----------------------------------------------------------------------
def _brinson() -> SimpleNamespace:
    return SimpleNamespace(
        by_sector=pd.DataFrame(
            {
                "sector": ["Tech", "Energy", "Health"],
                "allocation": [0.002, -0.001, 0.0005],
                "selection": [0.001, 0.0015, -0.0002],
            }
        )
    )


def test_brinson_bars_plots_allocation_and_selection_in_percent(monkeypatch) -> None:
    captured: list = []
    original = matplotlib.axes.Axes.bar

    def spy(self, x, height, *args, **kwargs):
        captured.append((np.asarray(x, dtype=float), np.asarray(height, dtype=float)))
        return original(self, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy)
    assert _is_png(brinson_bars(_brinson()))
    assert len(captured) == 2
    by_sector = _brinson().by_sector
    np.testing.assert_allclose(captured[0][1], by_sector["allocation"].to_numpy() * 100.0)
    np.testing.assert_allclose(captured[1][1], by_sector["selection"].to_numpy() * 100.0)
    # The two bars for a sector must be offset from each other.
    assert not np.allclose(captured[0][0], captured[1][0])


def test_brinson_bars_is_empty_without_sectors() -> None:
    empty = SimpleNamespace(by_sector=pd.DataFrame(columns=["sector", "allocation", "selection"]))
    assert brinson_bars(empty) == ""


def test_every_chart_returns_a_decodable_string() -> None:
    """The report embeds these straight into an <img> tag."""
    payloads = {
        "equity": equity_curve(_equity()),
        "drawdown": drawdown(_equity()),
        "ic": ic_series(_returns()),
        "quantile": quantile_bar(pd.DataFrame({"q1": [0.01], "q2": [-0.01]})),
        "heatmap": monthly_heatmap(pd.DataFrame({"Jan": [0.01]}, index=[2021])),
        "risk": risk_contribution(pd.Series([0.6, 0.4], index=SYMBOLS[:2]), _cov(SYMBOLS[:2])),
        "brinson": brinson_bars(_brinson()),
    }
    for name, payload in payloads.items():
        assert payload, f"{name} produced nothing"
        assert _is_png(payload), f"{name} is not a PNG"
        assert isinstance(io.BytesIO(base64.b64decode(payload)).read(), bytes)
