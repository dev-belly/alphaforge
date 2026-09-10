"""Offline tests for the point-in-time fundamentals layer.

Joining a fundamental on its **fiscal period** instead of its **public release
date** is the single most expensive mistake in quant research: the backtest
looks spectacular and the strategy loses money, because the model was reading
numbers nobody knew yet. :class:`FundamentalView` exists to make that mistake
impossible, so these tests pin the contract rather than the shapes:

  * a statement is invisible before its ``report_date`` and visible from it on -
    and ``fiscal_period`` is never a join key;
  * the staleness guard retires a release once it is older than
    ``max_staleness_days``, while :meth:`FundamentalView.staleness` keeps
    reporting its true age as a diagnostic;
  * a ratio divides a point-in-time numerator by a point-in-time denominator
    (or by the market cap *of the signal date*), and a zero or infinite
    denominator becomes NaN rather than poisoning the cross-section;
  * :func:`market_cap_estimate` returns values in the **caller's row order** and
    never lets a missing cap shadow an older, valid one.

Everything is deterministic: no network, no randomness, no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.features import fundamentals as fundamentals_module
from alphaforge.features.fundamentals import DERIVED_FIELDS, FundamentalView, market_cap_estimate

DATES = pd.DatetimeIndex(
    [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-15",
    ]
)
SYMBOLS = pd.Index(["AAA", "BBB"])


def _mcap(value: float = 1.0e9, dates: pd.DatetimeIndex = DATES, symbols: pd.Index = SYMBOLS):
    return pd.DataFrame(value, index=dates, columns=symbols)


def _view(
    fund: pd.DataFrame | None,
    *,
    dates: pd.DatetimeIndex = DATES,
    symbols: pd.Index = SYMBOLS,
    market_cap: pd.DataFrame | None = None,
    max_staleness_days: int = 550,
) -> FundamentalView:
    return FundamentalView.build(
        fund,
        dates,
        symbols,
        _mcap(dates=dates, symbols=symbols) if market_cap is None else market_cap,
        max_staleness_days,
    )


def _statement(rows: list[tuple[str, str, float]], field: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "report_date": pd.to_datetime([r[1] for r in rows]),
            field: [r[2] for r in rows],
        }
    )


def _column(frame: pd.DataFrame, sym: str = "AAA") -> np.ndarray:
    return frame[sym].to_numpy(dtype=float)


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The project logger does not propagate to ``caplog``, so capture directly."""
    captured: list[str] = []
    monkeypatch.setattr(
        fundamentals_module.log, "warning", lambda msg, *a, **k: captured.append(str(msg))
    )
    return captured


# ----------------------------------------------------------------------
# The look-ahead contract
# ----------------------------------------------------------------------
def test_a_statement_is_invisible_before_its_report_date() -> None:
    """The headline contract: no field leaks backwards in time."""
    fund = _statement([("AAA", "2024-01-05", 100.0), ("AAA", "2024-01-10", 200.0)], "total_assets")
    got = _column(_view(fund).pit_panel("total_assets"))
    expected = [np.nan, np.nan, np.nan, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0]
    np.testing.assert_array_equal(got, expected)


def test_the_release_date_is_the_key_never_the_fiscal_period() -> None:
    """A Q4 statement filed in February must not be readable in January."""
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "fiscal_period": ["2023-12-31"],
            "report_date": pd.to_datetime(["2024-02-01"]),
            "total_assets": [100.0],
        }
    )
    got = _view(fund).pit_panel("total_assets")
    assert got.isna().all().all(), "a February filing leaked into January via fiscal_period"


def test_each_symbol_gets_its_own_release_calendar() -> None:
    fund = _statement([("AAA", "2024-01-03", 10.0), ("BBB", "2024-01-09", 20.0)], "total_assets")
    view = _view(fund)
    aaa = _column(view.pit_panel("total_assets"), "AAA")
    bbb = _column(view.pit_panel("total_assets"), "BBB")
    np.testing.assert_array_equal(aaa[:6], [np.nan, 10.0, 10.0, 10.0, 10.0, 10.0])
    np.testing.assert_array_equal(bbb[:5], [np.nan] * 5)
    assert bbb[5] == 20.0, "BBB's release never became visible"


def test_a_panel_carries_the_requested_index_and_columns() -> None:
    fund = _statement([("AAA", "2024-01-02", 1.0)], "total_assets")
    got = _view(fund).pit_panel("total_assets")
    assert got.index.equals(DATES)
    assert got.columns.equals(SYMBOLS)
    assert got.dtypes.eq(float).all()


# ----------------------------------------------------------------------
# Staleness
# ----------------------------------------------------------------------
def test_the_staleness_guard_retires_an_aged_out_release() -> None:
    fund = _statement([("AAA", "2024-01-02", 100.0)], "total_assets")
    got = _column(_view(fund, max_staleness_days=5).pit_panel("total_assets"))
    expected = [100.0, 100.0, 100.0, 100.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    np.testing.assert_array_equal(got, expected)


def test_the_default_window_keeps_a_fresh_annual_report() -> None:
    """550 days: a 424-day-old 10-K survives, a 576-day-old one does not."""
    dates = pd.DatetimeIndex(["2024-01-02", "2025-03-01", "2025-08-01"])
    fund = _statement([("AAA", "2024-01-02", 100.0)], "total_assets")
    got = _column(_view(fund, dates=dates).pit_panel("total_assets"))
    np.testing.assert_array_equal(got, [100.0, 100.0, np.nan])


def test_a_newer_release_resets_the_staleness_clock() -> None:
    fund = _statement([("AAA", "2024-01-02", 100.0), ("AAA", "2024-01-09", 200.0)], "total_assets")
    got = _column(_view(fund, max_staleness_days=5).pit_panel("total_assets"))
    # 01-08 is 6 days past the 01-02 release and ages out; 01-09 brings a new one.
    np.testing.assert_array_equal(got[:7], [100.0, 100.0, 100.0, 100.0, np.nan, 200.0, 200.0])


def test_staleness_reports_days_since_the_latest_release() -> None:
    fund = _statement([("AAA", "2024-01-02", 1.0), ("AAA", "2024-01-08", 2.0)], "total_assets")
    got = _column(_view(fund).staleness())
    # The 01-08 release resets the clock to zero rather than reading 6 days.
    np.testing.assert_array_equal(got, [0, 1, 2, 3, 0, 1, 2, 3, 4, 7])


def test_staleness_is_nan_before_the_first_release() -> None:
    fund = _statement([("AAA", "2024-01-08", 1.0)], "total_assets")
    got = _column(_view(fund).staleness())
    np.testing.assert_array_equal(got[:4], [np.nan] * 4)
    assert got[4] == 0.0


def test_staleness_keeps_reporting_past_the_guard_window() -> None:
    """It is a data-quality diagnostic, so it must not be clipped by the guard."""
    fund = _statement([("AAA", "2024-01-02", 1.0)], "total_assets")
    got = _column(_view(fund, max_staleness_days=2).staleness())
    assert got[-1] == 13.0, "the diagnostic was clipped by max_staleness_days"


def test_staleness_ignores_symbols_outside_the_universe() -> None:
    fund = _statement([("ZZZ", "2024-01-02", 1.0)], "total_assets")
    assert _view(fund).staleness().isna().all().all()


# ----------------------------------------------------------------------
# Missing / degenerate inputs
# ----------------------------------------------------------------------
def test_an_unknown_field_yields_an_all_nan_float_panel() -> None:
    fund = _statement([("AAA", "2024-01-02", 1.0)], "total_assets")
    got = _view(fund).pit_panel("not_a_field")
    assert got.shape == (len(DATES), len(SYMBOLS))
    assert got.isna().all().all()
    assert got.dtypes.eq(float).all()


def test_a_field_that_is_entirely_missing_yields_an_all_nan_panel() -> None:
    fund = _statement([("AAA", "2024-01-02", np.nan)], "total_assets")
    assert _view(fund).pit_panel("total_assets").isna().all().all()


def test_symbols_outside_the_universe_are_ignored() -> None:
    fund = _statement([("ZZZ", "2024-01-02", 1.0)], "total_assets")
    got = _view(fund).pit_panel("total_assets")
    assert got.columns.equals(SYMBOLS)
    assert got.isna().all().all()


def test_rows_without_a_report_date_are_dropped() -> None:
    fund = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "report_date": pd.to_datetime(["2024-01-02", None]),
            "total_assets": [100.0, 999.0],
        }
    )
    view = _view(fund)
    assert len(view.data) == 1
    assert view.data["total_assets"].iloc[0] == 100.0


def test_a_string_report_date_is_coerced_to_datetime() -> None:
    fund = pd.DataFrame({"symbol": ["AAA"], "report_date": ["2024-01-02"], "total_assets": [100.0]})
    view = _view(fund)
    assert pd.api.types.is_datetime64_any_dtype(view.data["report_date"])
    assert view.pit_panel("total_assets")["AAA"].iloc[0] == 100.0


def test_duplicate_release_dates_keep_the_last_row() -> None:
    """Same convention as the price panel: the later row wins."""
    fund = _statement([("AAA", "2024-01-02", 100.0), ("AAA", "2024-01-02", 200.0)], "total_assets")
    assert _column(_view(fund).pit_panel("total_assets"))[-1] == 200.0


@pytest.mark.parametrize("empty", [None, pd.DataFrame(columns=["symbol", "report_date"])])
def test_an_empty_input_builds_a_degenerate_view(empty, warnings: list[str]) -> None:
    view = _view(empty)
    assert view.data.empty
    assert view.available_fields() == []
    assert view.pit_panel("total_assets").isna().all().all()
    assert view.staleness().isna().all().all()
    assert any("No fundamentals" in w for w in warnings)


def test_the_degenerate_view_keeps_its_grid() -> None:
    view = _view(None)
    assert view.dates.equals(DATES)
    assert view.symbols.equals(SYMBOLS)


# ----------------------------------------------------------------------
# Derived statement items
# ----------------------------------------------------------------------
def test_derived_items_are_materialised_when_their_inputs_are_present() -> None:
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "operating_cashflow": [30.0],
            "capex": [10.0],
            "net_income": [25.0],
            "total_debt": [40.0],
        }
    )
    row = _view(fund).data.iloc[0]
    assert row["fcf"] == 20.0
    assert row["ocf_minus_ni"] == 5.0
    assert row["ni_minus_ocf"] == -5.0
    assert row["ev"] == pytest.approx(1.0e9 + 40.0)


def test_derived_items_are_skipped_when_their_inputs_are_absent() -> None:
    fund = _statement([("AAA", "2024-01-02", 1.0)], "total_assets")
    data = _view(fund).data
    for field in ("fcf", "ocf_minus_ni", "ni_minus_ocf", "ev"):
        assert field not in data.columns


def test_enterprise_value_does_not_require_book_equity() -> None:
    """``ebit_to_ev`` declares ``fundamental:ebit,total_debt,market_cap``.

    Deriving ``ev`` only when ``total_equity`` was also present made the factor
    silently all-NaN for any provider that reports debt without equity - and
    ``data_requirement`` is a declaration, never enforced, so nothing complained.
    """
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "ebit": [1.1e7],
            "total_debt": [1.0e8],
        }
    )
    view = _view(fund)
    assert "ev" in view.data.columns
    assert view.ratio("ebit_to_ev")["AAA"].iloc[0] == pytest.approx(0.01)


def test_a_missing_debt_falls_back_to_plain_market_cap() -> None:
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "total_debt": [np.nan],
        }
    )
    assert _view(fund).data["ev"].iloc[0] == pytest.approx(1.0e9)


def test_available_fields_excludes_keys_and_fiscal_period() -> None:
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "fiscal_period": ["2023-12-31"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "operating_cashflow": [30.0],
            "capex": [10.0],
            "total_assets": [500.0],
        }
    )
    got = _view(fund).available_fields()
    assert set(got) == {"operating_cashflow", "capex", "total_assets", "fcf"}
    assert not {"symbol", "report_date", "fiscal_period"} & set(got)


# ----------------------------------------------------------------------
# Ratios
# ----------------------------------------------------------------------
def test_a_ratio_divides_a_pit_numerator_by_a_pit_denominator() -> None:
    """Both legs age independently - 10/100 then 10/200."""
    fund = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "report_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"]),
            "total_equity": [100.0, np.nan, 200.0],
            "net_income": [np.nan, 10.0, np.nan],
        }
    )
    got = _column(_view(fund).ratio("roe"))
    np.testing.assert_allclose(
        got[:7], [np.nan, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05], rtol=0, atol=1e-12
    )


def test_an_ev_ratio_uses_the_signal_date_market_cap_plus_pit_debt() -> None:
    market_cap = pd.DataFrame(
        {"AAA": [100.0] * 5 + [200.0] * 5, "BBB": 1.0}, index=DATES, columns=SYMBOLS
    )
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "ebit": [15.0],
            "total_debt": [50.0],
        }
    )
    got = _column(_view(fund, market_cap=market_cap).ratio("ebit_to_ev"))
    np.testing.assert_allclose(got, [0.1] * 5 + [0.06] * 5, rtol=1e-12)


def test_a_market_cap_denominator_is_the_cap_of_the_signal_date() -> None:
    market_cap = pd.DataFrame(
        {"AAA": [100.0] * 5 + [200.0] * 5, "BBB": 1.0}, index=DATES, columns=SYMBOLS
    )
    fund = _statement([("AAA", "2024-01-02", 10.0)], "net_income")
    got = _column(_view(fund, market_cap=market_cap).ratio("earnings_yield"))
    np.testing.assert_allclose(got, [0.1] * 5 + [0.05] * 5, rtol=1e-12)


def test_a_missing_debt_does_not_void_an_ev_ratio() -> None:
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "ebit": [15.0],
            "total_debt": [np.nan],
        }
    )
    view = _view(fund, market_cap=_mcap(150.0))
    assert view.ratio("ebit_to_ev")["AAA"].iloc[0] == pytest.approx(0.1)


def test_a_zero_denominator_is_nan_not_infinity() -> None:
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "net_income": [10.0],
            "total_assets": [0.0],
        }
    )
    assert np.isnan(_column(_view(fund).ratio("roa"))).all()


def test_a_zero_market_cap_is_nan_not_infinity() -> None:
    fund = _statement([("AAA", "2024-01-02", 10.0)], "net_income")
    got = _column(_view(fund, market_cap=_mcap(0.0)).ratio("earnings_yield"))
    assert np.isnan(got).all()


def test_an_infinite_ratio_is_mapped_to_nan() -> None:
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "net_income": [1.0],
            "total_equity": [5e-324],
        }
    )
    got = _column(_view(fund).ratio("roe"))
    assert np.isnan(got).all(), "an overflowing ratio escaped as inf"


def test_an_unknown_ratio_name_raises_key_error() -> None:
    fund = _statement([("AAA", "2024-01-02", 1.0)], "total_assets")
    with pytest.raises(KeyError, match="Unknown derived fundamental field"):
        _view(fund).ratio("price_to_dreams")


def test_every_derived_field_resolves() -> None:
    """No ratio in the library may reference a field nobody can produce."""
    fund = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": pd.to_datetime(["2024-01-02"]),
            "net_income": [10.0],
            "revenue": [100.0],
            "gross_profit": [40.0],
            "ebit": [12.0],
            "operating_cashflow": [15.0],
            "capex": [5.0],
            "total_assets": [200.0],
            "total_equity": [100.0],
            "total_debt": [50.0],
        }
    )
    view = _view(fund)
    for name in DERIVED_FIELDS:
        got = view.ratio(name)
        assert not got.isna().all().all(), f"{name} is entirely NaN with every input present"


# ----------------------------------------------------------------------
# Coverage
# ----------------------------------------------------------------------
def test_coverage_is_the_fraction_of_the_universe_with_a_statement() -> None:
    fund = _statement([("AAA", "2024-01-02", 1.0), ("BBB", "2024-01-08", 1.0)], "total_assets")
    got = _view(fund).coverage()["coverage"].to_numpy(dtype=float)
    np.testing.assert_array_equal(got[:5], [0.5, 0.5, 0.5, 0.5, 1.0])


def test_coverage_without_a_balance_sheet_is_a_float_nan_column() -> None:
    """An object-dtype column of NaNs here quietly poisons downstream arithmetic."""
    fund = _statement([("AAA", "2024-01-02", 1.0)], "revenue")
    got = _view(fund).coverage()
    assert got.shape == (len(DATES), 1)
    assert got["coverage"].dtype == float
    assert got["coverage"].isna().all()


# ----------------------------------------------------------------------
# market_cap_estimate
# ----------------------------------------------------------------------
def test_the_estimate_follows_the_callers_row_order() -> None:
    """Regression: ``merge_asof`` needs a report-date sort, the caller needs it undone.

    ``build`` orders rows by ``(symbol, report_date)`` and assigns the result
    straight back, so AAA (reporting 01-08) must not receive BBB's cap just
    because BBB reported earlier.
    """
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "report_date": pd.to_datetime(["2024-01-08", "2024-01-03"]),
        }
    )
    market_cap = pd.DataFrame({"AAA": 100.0, "BBB": 1_000.0}, index=DATES, columns=SYMBOLS)
    got = market_cap_estimate(df, market_cap).tolist()
    assert got == [100.0, 1000.0], f"row order was lost: {got}"


def test_the_estimate_is_matched_per_symbol() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "report_date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
        }
    )
    market_cap = pd.DataFrame({"AAA": 100.0, "BBB": 1_000.0}, index=DATES, columns=SYMBOLS)
    assert market_cap_estimate(df, market_cap).tolist() == [100.0, 1000.0]


def test_a_missing_cap_does_not_shadow_an_older_valid_one() -> None:
    """``stack`` keeps NaNs and ``merge_asof`` takes the nearest preceding row.

    Without dropping them first, a missing cap would mask an older, good one.
    """
    df = pd.DataFrame({"symbol": ["AAA"], "report_date": pd.to_datetime(["2024-01-05"])})
    market_cap = pd.DataFrame(
        {"AAA": [100.0, np.nan, np.nan, np.nan, np.nan]}, index=DATES[:5], columns=["AAA"]
    )
    assert market_cap_estimate(df, market_cap).tolist() == [100.0]


def test_a_cap_older_than_the_tolerance_is_not_carried_forward() -> None:
    market_cap = pd.DataFrame(
        {"AAA": [100.0]}, index=pd.DatetimeIndex(["2024-01-02"]), columns=["AAA"]
    )
    near = pd.DataFrame({"symbol": ["AAA"], "report_date": pd.to_datetime(["2024-01-05"])})
    far = pd.DataFrame({"symbol": ["AAA"], "report_date": pd.to_datetime(["2024-01-15"])})
    assert market_cap_estimate(near, market_cap).tolist() == [100.0]
    assert np.isnan(market_cap_estimate(far, market_cap)).all()


def test_no_usable_cap_at_all_returns_nan() -> None:
    df = pd.DataFrame({"symbol": ["AAA"], "report_date": pd.to_datetime(["2024-01-05"])})
    market_cap = pd.DataFrame(np.nan, index=DATES, columns=["AAA"])
    assert np.isnan(market_cap_estimate(df, market_cap)).all()


def test_an_empty_market_cap_frame_returns_nan_instead_of_raising() -> None:
    """A column-less frame made ``stack`` emit an int-typed key and crash the merge."""
    df = pd.DataFrame({"symbol": ["AAA"], "report_date": pd.to_datetime(["2024-01-05"])})
    for empty in (
        pd.DataFrame(index=DATES),
        pd.DataFrame(index=pd.DatetimeIndex([]), columns=["AAA"], dtype=float),
        pd.DataFrame(np.nan, index=DATES, columns=pd.Index([], dtype=object)),
    ):
        got = market_cap_estimate(df, empty)
        assert len(got) == 1
        assert np.isnan(got).all()


def test_an_empty_statement_frame_returns_an_empty_series() -> None:
    df = pd.DataFrame(
        {
            "symbol": pd.Series([], dtype=object),
            "report_date": pd.Series([], dtype="datetime64[ns]"),
        }
    )
    assert market_cap_estimate(df, _mcap()).empty
