"""Offline tests for the report's formatting and section helpers.

The report is the artefact a reader actually looks at, so the failures here are
about what the page *says*, not about whether it renders. Two of them matter:

  * **a missing number must not read as a measurement.** The module already
    treats ``None`` as "-", but ``NaN`` - which is what every unmeasured metric
    actually is, e.g. ``calmar`` with no drawdown or the IC statistics of a
    constant factor - was formatted straight through. The shipped sample report
    had ten cells reading "nan", including a whole constant factor's IC row and
    the specific-risk row's exposure. ``_fmt``, ``_pct`` and ``_table`` now
    render anything non-finite as "-", consistent with ``None`` and with
    ``_finite``, which the module already used as its "is this usable" test.
  * **a table cell must not be able to inject markup.** String cells and column
    headers are escaped; a factor or symbol name is data, not HTML.

``build_html`` itself is covered by the integration tests, which render the real
report; these tests pin the pieces it is assembled from.

Everything is deterministic: no network, no files written outside tmp.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.reporting import report as report_module
from alphaforge.reporting.report import (
    ReportInputs,
    _attr_table,
    _avg_weights,
    _cov_from_risk,
    _finite,
    _fmt,
    _gross_net_table,
    _img,
    _is_empty,
    _kv_table,
    _pct,
    _quantile_from_model,
    _regime_section,
    _safe_monthly,
    _stress_section,
    _table,
    build_html,
    write_report,
)


# ----------------------------------------------------------------------
# _fmt
# ----------------------------------------------------------------------
def test_fmt_renders_a_plain_number_to_four_places() -> None:
    assert _fmt(0.123456) == "0.1235"
    assert _fmt(-1.5) == "-1.5000"


def test_fmt_collapses_a_negligible_number_to_zero() -> None:
    """A rounding artefact must not print as 1e-09."""
    assert _fmt(1e-9) == "0.0000"
    assert _fmt(-1e-9) == "0.0000"


def test_fmt_switches_to_significant_digits_at_the_extremes() -> None:
    assert _fmt(1.5e7) == "1.5e+07"
    assert _fmt(1e-5) == "1e-05"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fmt_renders_a_non_finite_number_as_missing(value: float) -> None:
    """Regression: these used to print as "nan" / "inf" on the page."""
    assert _fmt(value) == "-"


def test_fmt_renders_none_as_missing() -> None:
    assert _fmt(None) == "-"


def test_fmt_passes_other_types_through_as_text() -> None:
    assert _fmt("ridge") == "ridge"
    assert _fmt(7) == "7"
    assert _fmt(True) == "True"


# ----------------------------------------------------------------------
# _pct
# ----------------------------------------------------------------------
def test_pct_scales_by_a_hundred() -> None:
    assert _pct(0.1234) == "12.34%"
    assert _pct(-0.005) == "-0.50%"
    assert _pct(0.0) == "0.00%"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_pct_renders_a_missing_value_as_a_dash(value) -> None:
    """Regression: a NaN percentage used to print as "nan%"."""
    assert _pct(value) == "-"


def test_pct_falls_back_to_fmt_for_a_non_number() -> None:
    assert _pct("n/a") == "n/a"


# ----------------------------------------------------------------------
# _finite
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value", [1, 0.5, -3.0, np.float64(2.0)])
def test_finite_accepts_real_numbers(value) -> None:
    assert _finite(value) is True


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "abc", object()])
def test_finite_rejects_anything_not_usable(value) -> None:
    assert _finite(value) is False


# ----------------------------------------------------------------------
# _table
# ----------------------------------------------------------------------
def test_an_empty_table_says_so() -> None:
    assert _table(pd.DataFrame()) == "<p><i>No data.</i></p>"
    assert _table(None) == "<p><i>No data.</i></p>"


def test_a_table_renders_headers_and_rows() -> None:
    got = _table(pd.DataFrame({"factor": ["mom"], "ic": [0.0123]}))
    assert "<th>factor</th>" in got and "<th>ic</th>" in got
    assert "<td>mom</td>" in got
    assert "<td>0.0123</td>" in got


def test_a_nan_cell_renders_as_a_dash() -> None:
    """Regression: the specific-risk row's exposure printed as "nan"."""
    got = _table(pd.DataFrame({"block": ["specific"], "exposure": [float("nan")]}))
    assert "<td>-</td>" in got
    assert "nan" not in got


def test_string_cells_and_headers_are_escaped() -> None:
    """A symbol or factor name is data, not markup."""
    got = _table(pd.DataFrame({"<script>x</script>": ["<b>bold</b>"]}))
    assert "<script>" not in got
    assert "&lt;script&gt;" in got
    assert "&lt;b&gt;bold&lt;/b&gt;" in got


def test_the_float_format_is_configurable() -> None:
    got = _table(pd.DataFrame({"x": [0.5]}), floatfmt="{:.1%}")
    assert "<td>50.0%</td>" in got


# ----------------------------------------------------------------------
# _kv_table / _gross_net_table
# ----------------------------------------------------------------------
def test_the_kv_table_percentages_only_the_percentage_keys() -> None:
    got = _kv_table({"cagr": 0.08, "sharpe": 0.9})
    assert "8.00%" in got
    assert "0.9000" in got
    assert "sharpe" in got


def test_the_kv_table_renders_a_missing_metric_as_a_dash() -> None:
    got = _kv_table({"cagr": float("nan"), "sharpe": None})
    assert "nan" not in got
    assert got.count("<td class='v'>-</td>") == 2


def test_the_gross_net_table_is_omitted_without_a_gross_cagr() -> None:
    assert _gross_net_table({}) == ""
    assert _gross_net_table({"cagr": 0.08}) == ""
    assert _gross_net_table({"gross_cagr": float("nan")}) == ""


def test_the_gross_net_table_reports_the_gap() -> None:
    got = _gross_net_table({"total_return": 0.20, "gross_total_return": 0.26, "gross_cagr": 0.05})
    assert "20.00%" in got and "26.00%" in got
    assert "6.00%" in got, "the gap is gross minus net"


def test_the_gross_net_table_skips_a_pair_that_is_not_reported() -> None:
    """A metric the backtest never produced must not become a row of dashes."""
    got = _gross_net_table({"gross_cagr": 0.05, "cagr": 0.04})
    assert "CAGR" in got
    assert "Sortino" not in got


def test_the_cost_drag_row_is_shown_when_known() -> None:
    got = _gross_net_table({"gross_cagr": 0.05, "cagr": 0.04, "cost_drag_cagr": 0.01})
    assert "Cost drag" in got and "1.00%" in got


def test_the_cost_drag_row_is_omitted_when_unknown() -> None:
    got = _gross_net_table({"gross_cagr": 0.05, "cagr": 0.04, "cost_drag_cagr": float("nan")})
    assert "Cost drag" not in got


# ----------------------------------------------------------------------
# _img and the small extractors
# ----------------------------------------------------------------------
def test_an_image_tag_needs_payload() -> None:
    assert _img("") == ""
    assert _img("QUJD") == '<img src="data:image/png;base64,QUJD" />'


def test_is_empty_handles_none_and_frames() -> None:
    assert _is_empty(None) is True
    assert _is_empty(pd.DataFrame()) is True
    assert _is_empty(pd.DataFrame({"a": [1]})) is False
    assert _is_empty("not a frame") is False


def test_safe_monthly_returns_an_empty_frame_on_failure(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ValueError("no")

    monkeypatch.setattr("alphaforge.backtest.metrics.monthly_returns", boom)
    assert _safe_monthly(pd.Series([0.01])).empty


def test_safe_monthly_passes_a_normal_series_through() -> None:
    r = pd.Series(0.01, index=pd.date_range("2020-01-31", periods=12, freq="ME"))
    got = _safe_monthly(r)
    assert not got.empty


def test_avg_weights_is_the_mean_absolute_weight() -> None:
    weights = pd.DataFrame({"A": [0.2, -0.4], "B": [0.0, 0.6]})
    got = _avg_weights(weights)
    assert got["A"] == pytest.approx(0.3)
    assert got["B"] == pytest.approx(0.3)


def test_quantile_returns_are_extracted_from_the_model_summary() -> None:
    table = pd.DataFrame({"mean": [0.01]})
    got = _quantile_from_model(ReportInputs(model_summary={"quantile_returns": table}))
    assert got is table


def test_quantile_returns_are_empty_without_a_model_summary() -> None:
    assert _quantile_from_model(ReportInputs()).empty
    assert _quantile_from_model(ReportInputs(model_summary={})).empty


def test_the_covariance_is_read_off_the_decomposition_attrs() -> None:
    cov = pd.DataFrame(np.eye(2))
    table = pd.DataFrame({"a": [1]})
    table.attrs["covariance"] = cov
    assert _cov_from_risk(ReportInputs(risk_decomposition=table)) is cov


def test_the_covariance_is_none_when_absent() -> None:
    assert _cov_from_risk(ReportInputs()) is None
    assert _cov_from_risk(ReportInputs(risk_decomposition=pd.DataFrame({"a": [1]}))) is None


# ----------------------------------------------------------------------
# _regime_section
# ----------------------------------------------------------------------
def test_the_regime_section_is_empty_without_a_regime() -> None:
    assert _regime_section(None, None) == ""
    assert _regime_section(pd.Series(dtype=object), None) == ""
    assert _regime_section(pd.Series([np.nan, np.nan]), None) == ""


def test_the_regime_section_is_empty_for_a_non_series() -> None:
    """A plain dict is not a per-day regime and must not be iterated as one."""
    assert _regime_section({"Bull": 10}, None) == ""


def test_the_regime_section_counts_the_days() -> None:
    regime = pd.Series(["Bull"] * 3 + ["Bear"])
    got = _regime_section(regime, None)
    assert "Market Regime" in got
    assert "Bull" in got and "75%" in got
    assert "Bear" in got and "25%" in got


def test_the_regime_section_renders_the_stats_table() -> None:
    regime = pd.Series(["Bull"] * 2)
    stats = {"Bull": {"n_days": 2, "ann_return": 0.12, "ann_vol": 0.18, "sharpe": 0.9}}
    got = _regime_section(regime, stats)
    assert "12.00%" in got and "18.00%" in got
    assert "0.9000" in got


def test_the_regime_stats_table_renders_a_missing_metric_as_a_dash() -> None:
    got = _regime_section(pd.Series(["Bull"]), {"Bull": {"n_days": 1}})
    assert "nan" not in got


# ----------------------------------------------------------------------
# _stress_section
# ----------------------------------------------------------------------
def _stress_result(pnl: float, contributions: pd.Series | None = None):
    from alphaforge.risk.stress import StressResult

    return StressResult(
        scenario="s", pnl_pct=pnl, shock={"market": -0.1}, contributions=contributions
    )


def test_the_stress_section_is_empty_without_results() -> None:
    assert _stress_section(None) == ""
    assert _stress_section({}) == ""


def test_the_stress_section_lists_the_scenarios_and_their_worst_holdings() -> None:
    contributions = pd.Series({"A": -0.05, "B": -0.02, "C": 0.01})
    got = _stress_section({"crash": _stress_result(-0.06, contributions)})
    assert "Stress Testing" in got
    assert "crash" in got
    assert "-6.00%" in got
    assert "A -5.00%" in got and "B -2.00%" in got


def test_a_scenario_without_contributions_shows_a_dash() -> None:
    got = _stress_section({"crash": _stress_result(-0.06)})
    assert "<td>-</td>" in got


# ----------------------------------------------------------------------
# _attr_table
# ----------------------------------------------------------------------
def test_the_attribution_table_needs_an_input() -> None:
    assert _attr_table(None).empty


def test_the_attribution_table_is_built_from_the_betas_index() -> None:
    fa = SimpleNamespace(
        betas=pd.Series({"value": 0.4, "momentum": -0.1}),
        attributed_return=pd.Series({"value": 0.002, "momentum": -0.001}),
        t_stats=pd.Series({"value": 2.1, "momentum": -0.8}),
    )
    got = _attr_table(fa)
    assert list(got["factor"]) == ["value", "momentum"]
    assert list(got["beta"]) == [0.4, -0.1]
    assert list(got["t_stat"]) == [2.1, -0.8]


def test_the_attribution_table_aligns_to_the_betas_index() -> None:
    """A series in a different order must be reindexed, not zipped blindly."""
    fa = SimpleNamespace(
        betas=pd.Series({"value": 0.4, "momentum": -0.1}),
        attributed_return=pd.Series({"momentum": -0.001, "value": 0.002}),
        t_stats=pd.Series({"momentum": -0.8, "value": 2.1}),
    )
    got = _attr_table(fa)
    assert list(got["factor"]) == ["value", "momentum"]
    assert list(got["attributed_return"]) == [0.002, -0.001]


# ----------------------------------------------------------------------
# build_html / write_report
# ----------------------------------------------------------------------
def test_build_html_renders_a_minimal_report() -> None:
    got = build_html(ReportInputs(title="T", notes=["hello"]))
    assert got.startswith("<!DOCTYPE html>")
    assert "<title>T</title>" in got
    assert "hello" in got


def test_build_html_falls_back_when_there_are_no_notes() -> None:
    assert "None." in build_html(ReportInputs())


def test_build_html_escapes_a_note() -> None:
    got = build_html(ReportInputs(notes=["<script>alert(1)</script>"]))
    assert "<script>alert(1)</script>" not in got
    assert "&lt;script&gt;" in got


def _backtest(**overrides):
    base = {
        "metrics": {"cagr": float("nan"), "sharpe": 0.5, "total_return": 0.1},
        "equity": pd.Series([1.0, 1.1], index=pd.date_range("2020-01-01", periods=2)),
        "returns": pd.Series([0.1], index=pd.date_range("2020-01-01", periods=1)),
        "weights": pd.DataFrame({"A": [0.5]}),
        "config": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_html_never_prints_a_missing_number_as_nan() -> None:
    """The end-to-end version of the formatting regression.

    Matched as the rendered *patterns* rather than the bare substring: the
    report embeds base64 PNG payloads, and those contain "nan" by chance.
    """
    got = build_html(ReportInputs(backtest=_backtest()))
    assert ">nan<" not in got
    assert "nan%" not in got
    assert ">inf<" not in got


def test_build_html_disclaims_synthetic_data() -> None:
    """A sample-provider report must say the results prove nothing."""
    got = build_html(ReportInputs(config={"data": {"provider": "sample"}}))
    assert "Synthetic data" in got
    assert "do not establish a tradeable edge" in got


def test_build_html_does_not_disclaim_a_real_provider() -> None:
    got = build_html(ReportInputs(config={"data": {"provider": "polygon"}}))
    assert "Synthetic data" not in got
    assert "polygon" in got


def test_build_html_renders_the_risk_contribution_chart_when_a_covariance_exists() -> None:
    cov = pd.DataFrame(np.eye(1), index=["A"], columns=["A"])
    decomposition = pd.DataFrame({"exposure": [0.5]})
    decomposition.attrs["covariance"] = cov
    got = build_html(ReportInputs(backtest=_backtest(), risk_decomposition=decomposition))
    assert "data:image/png;base64," in got


def test_build_html_renders_the_quantile_chart_from_the_model_summary() -> None:
    table = pd.DataFrame({"mean": [0.01, -0.01]})
    got = build_html(ReportInputs(model_summary={"quantile_returns": table}))
    assert "data:image/png;base64," in got


def test_build_html_renders_the_brinson_chart() -> None:
    brinson = SimpleNamespace(
        by_sector=pd.DataFrame({"sector": ["Tech"], "allocation": [0.001], "selection": [-0.001]})
    )
    got = build_html(ReportInputs(brinson=brinson))
    assert "data:image/png;base64," in got


def test_covariance_lookup_survives_an_object_without_attrs() -> None:
    """The broad except is what keeps a malformed input from killing the report."""
    assert _cov_from_risk(ReportInputs(risk_decomposition=object())) is None


def test_write_report_creates_the_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "report.html"
    got = write_report(ReportInputs(title="X"), target)
    assert got == target
    assert target.exists()
    assert "<title>X</title>" in target.read_text(encoding="utf-8")


def test_write_report_overwrites_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "report.html"
    target.write_text("stale")
    write_report(ReportInputs(title="Fresh"), target)
    assert "Fresh" in target.read_text(encoding="utf-8")
    assert "stale" not in target.read_text(encoding="utf-8")


def test_the_module_documents_its_public_surface() -> None:
    assert set(report_module.__all__) == {"ReportInputs", "build_html", "write_report"}


def test_finite_and_fmt_agree_about_what_is_missing() -> None:
    """The two helpers must not disagree, or a guard and its renderer diverge."""
    for value in (None, float("nan"), float("inf"), float("-inf")):
        assert not _finite(value)
        assert _fmt(value) == "-"
    for value in (0.0, 1.0, -1.0, 1e9):
        assert _finite(value)
        assert _fmt(value) != "-"
    assert math.isfinite(0.0)
