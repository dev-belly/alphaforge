"""Report output keeps metric units and the data source visible to readers."""

from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.reporting import charts
from alphaforge.reporting.report import ReportInputs, build_html


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag == "td":
            self.in_cell = True

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_cell = False
        if tag == "tr":
            self.rows.append(self.row)

    def handle_data(self, data):
        if self.in_cell:
            self.row.append(data)


def test_report_preserves_dimensionless_ratios():
    bt = SimpleNamespace(
        equity=pd.Series(dtype=float),
        returns=pd.Series(dtype=float),
        metrics={
            "cagr": 0.12,
            "sharpe": 1.25,
            "sortino": 1.75,
            "calmar": 0.6,
            "information_ratio": 0.35,
        },
    )
    rows = _Rows()
    rows.feed(build_html(ReportInputs(backtest=bt)))
    values = {row[0]: row[1] for row in rows.rows if len(row) == 2}
    assert values["cagr"] == "12.00%"
    assert values["sharpe"] == "1.2500"
    assert values["sortino"] == "1.7500"
    assert values["calmar"] == "0.6000"
    assert values["information_ratio"] == "0.3500"


@pytest.mark.parametrize(
    ("provider", "label", "synthetic"),
    [
        ("sample", "sample", True),
        ("synthetic", "synthetic", True),
        (" SAMPLE ", "sample", True),
        (None, "sample", True),
        ("", "sample", True),
        ("local", "local", False),
        ("yahoo", "yahoo", False),
    ],
)
def test_report_discloses_data_source_before_performance(provider, label, synthetic):
    report = build_html(ReportInputs(config={"data": {"provider": provider}}))
    assert report.index(f"Data provider: {label}.") < report.index("<h2>Performance</h2>")
    assert ("Synthetic data" in report) is synthetic


def test_equity_chart_compounds_benchmark_from_initial_cash(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    equity = pd.Series([99.0, 101.0, 102.0], index=dates)
    # Returns outside the backtest window must not influence plotted wealth.
    benchmark = pd.Series([0.5, 0.10, -0.05, 0.02], index=[dates[0] - pd.Timedelta(days=1), *dates])
    captured = {}

    def capture(fig):
        captured.update({line.get_label(): line.get_ydata() for line in fig.axes[0].lines})
        charts.plt.close(fig)
        return "figure"

    monkeypatch.setattr(charts, "_fig_to_b64", capture)
    charts.equity_curve(equity, benchmark, initial_capital=100.0)
    np.testing.assert_allclose(captured["Benchmark"], [110.0, 104.5, 106.59])
    np.testing.assert_array_equal(captured["Net (after-cost)"], equity.to_numpy())


def test_equity_chart_preserves_missing_benchmark_sessions(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    equity = pd.Series([100.0, 100.0, 100.0], index=dates)
    benchmark = pd.Series([0.10, -0.05], index=dates[[0, 2]])
    captured = {}

    def capture(fig):
        line = next(line for line in fig.axes[0].lines if line.get_label() == "Benchmark")
        captured["y"] = line.get_ydata()
        captured["x"] = line.get_xdata()
        charts.plt.close(fig)
        return "figure"

    monkeypatch.setattr(charts, "_fig_to_b64", capture)
    charts.equity_curve(equity, benchmark, initial_capital=100.0)
    np.testing.assert_allclose(captured["y"], [110.0, np.nan, 104.5], equal_nan=True)
    np.testing.assert_array_equal(captured["x"], dates.to_numpy())
