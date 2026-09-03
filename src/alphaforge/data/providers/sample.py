"""Deterministic **synthetic** market-data generator.

.. warning::
   The data produced here is *simulated*, not real market history. It exists so
   that a fresh clone can run the entire platform end-to-end without any API
   key, and so that the research engine can be validated against a known ground
   truth (the factor premia explicitly embedded below).

   Every artefact written by this provider is labelled
   ``provenance = "SYNTHETIC_SAMPLE"`` and the platform refuses to present it
   as real market data.

Embedded (known) structure
--------------------------
* one market factor, 11 sector factors and five style factors
  (size, value, momentum, low-volatility, quality) with non-zero premia;
* a persistent idiosyncratic drift component (→ medium-term momentum) plus a
  transient bid-ask-bounce component (→ short-horizon reversal).  Serial
  correlation, not a "momentum premium", is the mechanism that produces
  momentum in the cross-section - which is how it works in real markets;
* heterogeneous idiosyncratic volatility with a negative volatility premium
  (the low-volatility anomaly);
* latent style exposures **shared between the price process and the
  fundamental statements**, so value and quality signals derived from
  financials genuinely carry information about future returns;
* corporate actions (splits, dividends) so ``close`` != ``adj_close``;
* **delistings**, so point-in-time index membership is not survivorship-free
  and the platform's survivorship handling is actually exercised;
* quarterly fundamentals released with a 45-110 day publication lag measured
  from the fiscal period end.  The pipeline keys off ``report_date``, never off
  ``fiscal_period``.

Calibration
-----------
The noise/premium split is calibrated so that realised information coefficients
land in the range reported in the empirical literature (|IC| ~ 0.02-0.05 per
21-day holding period) rather than the absurd values a naive simulation yields.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.data.providers.base import (
    CONSTITUENT_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    GICS_SECTORS,
    MACRO_COLUMNS,
    PRICE_COLUMNS,
    DataProvider,
)
from alphaforge.utils.logging import get_logger

log = get_logger("data.sample")

PROVENANCE = "SYNTHETIC_SAMPLE"

# Annualised premia per unit of standardised exposure (ground truth).
EMBEDDED_PREMIA: dict[str, float] = {
    "market": 0.070,
    "size": -0.015,  # small-cap tilt earns a premium => negative size loading
    "value": 0.055,
    "momentum": 0.060,
    "low_volatility": 0.045,
    "quality": 0.060,
}

# Cross-sectional dispersion of the standardised style exposures.
EXPOSURE_SCALE = 0.40

# Volatility regimes the market factor switches between (annualised).
VOL_REGIMES = (0.08, 0.13, 0.24)

# Idiosyncratic return calibration.  ``DRIFT_SD`` is the stationary standard
# deviation of the persistent component and is the *only* knob controlling the
# strength of medium-term momentum; ``TRANSIENT_SD`` plus ``BOUNCE_THETA``
# control short-horizon reversal.
DRIFT_RHO = 0.97
DRIFT_SD = 0.00050
TRANSIENT_SD = 0.007
BOUNCE_THETA = 0.40
# Expected-return penalty per unit of idiosyncratic-volatility z-score (the
# low-volatility anomaly), expressed as an annualised drift.
VOL_ANOMALY = 0.045


@dataclass
class SampleSpec:
    n_symbols: int = 160
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    seed: int = 42
    delisting_rate: float = 0.10
    index_id: str = "SP500_SAMPLE"


class SampleDataProvider(DataProvider):
    """Generates a reproducible cross-section of synthetic equity data."""

    name = "sample"

    def __init__(self, spec: SampleSpec | None = None) -> None:
        self.spec = spec or SampleSpec()
        self._cache: dict[str, object] = {}
        self._latent: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def symbols(self) -> list[str]:
        return [f"SIM{i:04d}" for i in range(1, self.spec.n_symbols + 1)]

    def calendar(self) -> pd.DatetimeIndex:
        return pd.bdate_range(self.spec.start, self.spec.end, name="date")

    def fetch_prices(
        self, symbols: Sequence[str] | None = None, start=None, end=None
    ) -> pd.DataFrame:
        if "prices" not in self._cache:
            self._cache["prices"] = self._simulate_prices()
        return self._cache["prices"].copy()  # type: ignore[attr-defined]

    def fetch_fundamentals(
        self, symbols: Sequence[str] | None = None, start=None, end=None
    ) -> pd.DataFrame:
        if "fundamentals" not in self._cache:
            self._cache["fundamentals"] = self._simulate_fundamentals()
        return self._cache["fundamentals"].copy()  # type: ignore[attr-defined]

    def fetch_constituents(self, index_id: str | None = None, start=None, end=None) -> pd.DataFrame:
        if "constituents" not in self._cache:
            self._cache["constituents"] = self._simulate_constituents()
        return self._cache["constituents"].copy()  # type: ignore[attr-defined]

    def fetch_macro(
        self, series: Sequence[str] | None = None, start=None, end=None
    ) -> pd.DataFrame:
        if "macro" not in self._cache:
            self._cache["macro"] = self._simulate_macro()
        return self._cache["macro"].copy()  # type: ignore[attr-defined]

    def fetch_industry(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        if "industry" not in self._cache:
            prices = self.fetch_prices()
            self._cache["industry"] = prices[["symbol", "industry"]].drop_duplicates("symbol")
        return self._cache["industry"].copy()  # type: ignore[attr-defined]

    def benchmark_prices(self, index_id: str | None = None, start=None, end=None) -> pd.Series:
        if "benchmark" in self._cache:
            return self._cache["benchmark"].copy()  # type: ignore[attr-defined]
        prices = self.fetch_prices()
        close = prices.pivot(index="date", columns="symbol", values="adj_close")
        mcap = prices.pivot(index="date", columns="symbol", values="market_cap")
        rets = close.pct_change(fill_method=None)
        w = mcap.shift(1)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
        idx_ret = (w * rets).sum(axis=1, min_count=1).fillna(0.0)
        bench = 100.0 * (1.0 + idx_ret).cumprod()
        bench.name = "benchmark"
        self._cache["benchmark"] = bench
        return bench.copy()

    # ------------------------------------------------------------------
    # shared latent structure
    # ------------------------------------------------------------------
    def _build_latent(self) -> dict[str, np.ndarray]:
        """Latent, asset-level attributes shared by prices and fundamentals."""
        if self._latent is not None:
            return self._latent
        rng = np.random.default_rng(self.spec.seed)
        n = self.spec.n_symbols
        lat = {
            "industry_idx": rng.integers(0, len(GICS_SECTORS), size=n),
            "industry": np.array(
                [GICS_SECTORS[i] for i in rng.integers(0, len(GICS_SECTORS), size=n)], dtype=object
            ),
            "beta": np.clip(1.00 + 0.30 * rng.standard_normal(n), 0.25, 2.20),
            "log_mcap": 23.2 + 1.35 * rng.standard_normal(n),
            "value_score": EXPOSURE_SCALE * rng.standard_normal(n),
            "quality_score": EXPOSURE_SCALE * rng.standard_normal(n),
            "momentum_score": EXPOSURE_SCALE * rng.standard_normal(n),
            "lowvol_score": EXPOSURE_SCALE * rng.standard_normal(n),
            "idio_vol": np.clip(0.90 + 0.55 * rng.standard_normal(n), 0.25, 3.50),
            "pb_base": np.exp(rng.normal(np.log(3.0), 0.45, size=n)),
            "roe_base": np.clip(rng.normal(0.14, 0.055, size=n), -0.05, 0.45),
            "gross_margin_base": np.clip(rng.normal(0.36, 0.12, size=n), 0.06, 0.85),
            "leverage": np.clip(rng.normal(0.32, 0.14, size=n), 0.05, 0.78),
            "asset_turnover": np.clip(rng.normal(1.0, 0.32, size=n), 0.20, 2.60),
        }
        lat["industry"] = np.array([GICS_SECTORS[i] for i in lat["industry_idx"]], dtype=object)
        lat["size_score"] = (lat["log_mcap"] - lat["log_mcap"].mean()) / lat["log_mcap"].std()
        lat["vol_score"] = (lat["idio_vol"] - lat["idio_vol"].mean()) / lat["idio_vol"].std()
        self._latent = lat
        return lat

    # ------------------------------------------------------------------
    # price simulation
    # ------------------------------------------------------------------
    def _simulate_prices(self) -> pd.DataFrame:
        lat = self._build_latent()
        rng = np.random.default_rng(self.spec.seed + 7)
        dates = self.calendar()
        n_dates, n_assets = len(dates), self.spec.n_symbols
        symbols = self.symbols()
        dt = 1.0 / 252.0
        sqrt_dt = np.sqrt(dt)
        mu = EMBEDDED_PREMIA

        # ---- regime-switching market volatility --------------------------
        vol_state = np.zeros(n_dates)
        state = VOL_REGIMES[1]
        for t in range(n_dates):
            if rng.random() < 0.004:
                state = VOL_REGIMES[rng.integers(0, len(VOL_REGIMES))]
            vol_state[t] = state
        f_market = rng.normal(0.0, 1.0, n_dates) * vol_state * sqrt_dt + mu["market"] * dt

        def factor_path(annual_mu: float, annual_vol: float) -> np.ndarray:
            sigma = annual_vol * (vol_state / VOL_REGIMES[1])
            return rng.normal(0.0, 1.0, n_dates) * sigma * sqrt_dt + annual_mu * dt

        f_size = factor_path(mu["size"], 0.09)
        f_value = factor_path(mu["value"], 0.10)
        f_momentum = factor_path(mu["momentum"], 0.12)
        f_lowvol = factor_path(mu["low_volatility"], 0.09)
        f_quality = factor_path(mu["quality"], 0.08)

        sector_paths = {
            s: factor_path(float(rng.normal(0.02, 0.025)), float(rng.uniform(0.05, 0.11)))
            for s in GICS_SECTORS
        }
        sector_ret = np.column_stack([sector_paths[s] for s in lat["industry"]])

        # ---- idiosyncratic returns ---------------------------------------
        # Persistent AR(1) drift => medium-term momentum.
        shock_sd = DRIFT_SD * np.sqrt(1.0 - DRIFT_RHO**2)
        drift = np.zeros((n_dates, n_assets))
        for t in range(1, n_dates):
            drift[t] = DRIFT_RHO * drift[t - 1] + rng.normal(0, shock_sd, n_assets)
        # Transient component with negative autocorrelation => short-term reversal.
        transient = rng.normal(0, TRANSIENT_SD, size=(n_dates, n_assets))
        bounce = np.zeros_like(transient)
        bounce[1:] = -BOUNCE_THETA * transient[:-1]
        idio = (
            drift
            + transient
            + bounce
            # Low-volatility anomaly: high-vol names earn less per unit of risk.
            - VOL_ANOMALY * dt * lat["vol_score"][None, :]
        )

        rets = (
            lat["beta"][None, :] * f_market[:, None]
            + lat["size_score"][None, :] * f_size[:, None]
            + lat["value_score"][None, :] * f_value[:, None]
            + lat["momentum_score"][None, :] * f_momentum[:, None]
            + lat["quality_score"][None, :] * f_quality[:, None]
            + lat["lowvol_score"][None, :] * f_lowvol[:, None]
            + sector_ret
            + lat["idio_vol"][None, :] * idio
        )

        # ---- delistings: a large terminal jump, then no further prints ----
        alive = np.ones((n_dates, n_assets), dtype=bool)
        n_delist = int(self.spec.delisting_rate * n_assets)
        for j in rng.choice(n_assets, size=n_delist, replace=False):
            cut = int(rng.integers(int(0.6 * n_dates), n_dates - 5))
            alive[cut:, j] = False
            rets[cut - 1, j] -= float(rng.uniform(0.35, 0.75))
        rets = np.where(alive, rets, np.nan)
        rets[0, :] = 0.0

        # ---- prices, splits and dividends --------------------------------
        adj_close = 100.0 * np.exp(np.cumsum(np.nan_to_num(rets, nan=0.0), axis=0))
        adj_close[~alive] = np.nan

        split_adj = np.ones((n_dates, n_assets))
        ratio_cum = np.ones(n_assets)
        for j in range(n_assets):
            for _ in range(rng.integers(0, 2)):
                s = int(rng.integers(30, n_dates - 1))
                ratio = float(rng.choice([2.0, 3.0]))
                split_adj[:s, j] /= ratio
                ratio_cum[j] *= ratio

        div_yield = np.clip(0.012 + 0.010 * rng.standard_normal(n_assets), 0.0, 0.06)
        div_adj = np.exp(np.cumsum(div_yield[None, :] * dt, axis=0))
        close = adj_close / (split_adj * div_adj)

        shares_adj = np.exp(lat["log_mcap"]) / 100.0 * ratio_cum
        shares_out = shares_adj[None, :] * split_adj
        market_cap = adj_close * shares_adj[None, :]

        dollar_vol = (
            np.exp(17.0 + 0.85 * rng.standard_normal(n_assets))[None, :]
            * (1.0 + 6.0 * np.abs(np.nan_to_num(rets)))
            * np.exp(0.4 * rng.normal(0, 0.5, size=(n_dates, n_assets)))
        )
        volume = np.round(dollar_vol / np.maximum(close, 1e-6))
        volume[~alive] = 0

        intraday = np.abs(rng.normal(0, 0.008, size=(n_dates, n_assets)))
        open_ = close * (1.0 + rng.normal(0, 0.004, size=(n_dates, n_assets)))
        high = np.maximum(open_, close) * (1.0 + intraday)
        low = np.minimum(open_, close) * (1.0 - intraday)

        frames = [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": open_[:, j],
                    "high": high[:, j],
                    "low": low[:, j],
                    "close": close[:, j],
                    "adj_close": adj_close[:, j],
                    "volume": volume[:, j],
                    "market_cap": market_cap[:, j],
                    "shares_outstanding": shares_out[:, j],
                    "industry": lat["industry"][j],
                }
            )
            for j, sym in enumerate(symbols)
        ]
        out = pd.concat(frames, ignore_index=True)
        out = out[np.isfinite(out["adj_close"]) & (out["adj_close"] > 0)]
        out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
        out.attrs["provenance"] = PROVENANCE
        log.info(
            f"Simulated synthetic panel: {len(out):,} rows | {out['symbol'].nunique()} symbols | "
            f"{out['date'].nunique()} dates | {n_delist} delistings"
        )
        return out[PRICE_COLUMNS]

    # ------------------------------------------------------------------
    # fundamentals simulation (shares the latent exposures)
    # ------------------------------------------------------------------
    def _simulate_fundamentals(self) -> pd.DataFrame:
        lat = self._build_latent()
        rng = np.random.default_rng(self.spec.seed + 101)
        prices = self.fetch_prices()
        mcap_panel = prices.pivot_table(
            index="date", columns="symbol", values="market_cap", aggfunc="last"
        )
        symbols = self.symbols()

        quarters = pd.period_range(self.spec.start, self.spec.end, freq="Q")
        rows = []
        for j, sym in enumerate(symbols):
            if sym not in mcap_panel.columns:
                continue
            mcap_by_q = mcap_panel[sym].reindex(
                pd.DatetimeIndex([q.end_time.normalize() for q in quarters]), method="ffill"
            )
            for q, period in enumerate(quarters):
                mcap = (
                    float(mcap_by_q.iloc[q])
                    if np.isfinite(mcap_by_q.iloc[q])
                    else float(np.exp(lat["log_mcap"][j]))
                )
                # --- value channel: book-to-price loads on value_score ------
                pb = lat["pb_base"][j] * float(
                    np.exp(-(lat["value_score"][j] + 0.35 * rng.normal()))
                )
                pb = float(np.clip(pb, 0.35, 40.0))
                book_equity = mcap / pb
                # --- quality channel: profitability loads on quality_score ---
                roe = float(
                    np.clip(
                        lat["roe_base"][j] + 0.85 * lat["quality_score"][j] + 0.03 * rng.normal(),
                        -0.60,
                        0.95,
                    )
                )
                net_income = roe * book_equity
                gross_margin = float(
                    np.clip(
                        lat["gross_margin_base"][j]
                        + 0.35 * lat["quality_score"][j]
                        + 0.02 * rng.normal(),
                        0.02,
                        0.95,
                    )
                )
                # Net margin implied by asset turnover and profitability.
                leverage = float(lat["leverage"][j])
                total_assets = book_equity / np.clip(1.0 - leverage, 0.12, 0.97)
                revenue = total_assets * lat["asset_turnover"][j] * (1.0 + 0.004 * q)
                revenue = float(np.maximum(revenue, 1e6))
                cogs = revenue * (1.0 - gross_margin)
                gross_profit = revenue - cogs
                net_margin = float(np.clip(net_income / revenue, -0.8, 0.6))
                ebit = net_income / np.clip(1.0 - 0.22, 0.35, 0.95)
                ocf = net_income + 0.03 * revenue * (0.5 + 0.5 * lat["quality_score"][j])
                capex = abs(0.045 * revenue + 0.01 * revenue * rng.normal())
                total_debt = total_assets - book_equity

                period_end = period.end_time.normalize()
                # Publication lag: 45-110 calendar days after the fiscal period end.
                report_date = period_end + pd.Timedelta(days=int(rng.integers(45, 111)))
                rows.append(
                    {
                        "symbol": sym,
                        "fiscal_period": str(period),
                        "period_end": period_end,
                        "report_date": report_date,
                        "revenue": revenue,
                        "cogs": cogs,
                        "gross_profit": gross_profit,
                        "ebit": float(ebit),
                        "net_income": float(net_income),
                        "total_assets": float(total_assets),
                        "total_equity": float(book_equity),
                        "total_debt": float(total_debt),
                        "operating_cashflow": float(ocf),
                        "capex": float(capex),
                        "industry": str(lat["industry"][j]),
                        "net_margin": net_margin,
                    }
                )
        df = pd.DataFrame(rows).sort_values(["symbol", "report_date"]).reset_index(drop=True)
        keep = [c for c in FUNDAMENTAL_COLUMNS if c in df.columns]
        out = df[keep + ["period_end", "industry"]]
        log.info(f"Simulated fundamentals: {len(out):,} rows | {out['symbol'].nunique()} symbols")
        return out

    # ------------------------------------------------------------------
    # index membership & macro
    # ------------------------------------------------------------------
    def _simulate_constituents(self) -> pd.DataFrame:
        prices = self.fetch_prices()
        mcap = prices.pivot_table(
            index="date", columns="symbol", values="market_cap", aggfunc="last"
        )
        mcap = mcap.resample("ME").last().ffill()
        rows = []
        for date, row in mcap.iterrows():
            live = row.dropna()
            if live.empty:
                continue
            w = live / live.sum()
            ordered = w.sort_values(ascending=False)
            keep = ordered[ordered.cumsum() <= 0.92].index
            if len(keep) < 20:
                keep = ordered.index[:20]
            w = w.loc[keep] / w.loc[keep].sum()
            for sym, weight in w.items():
                rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "index_id": self.spec.index_id,
                        "weight": float(weight),
                    }
                )
        df = pd.DataFrame(rows)[CONSTITUENT_COLUMNS]
        log.info(f"Simulated index membership: {len(df):,} rows | {df['date'].nunique()} snapshots")
        return df

    def _simulate_macro(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.spec.seed + 202)
        dates = pd.DatetimeIndex(sorted(set(self.calendar().to_period("M").to_timestamp("M"))))
        n = len(dates)

        def ar1(x0: float, kappa: float, sigma: float, mu: float) -> np.ndarray:
            out = np.zeros(n)
            x = x0
            for t in range(n):
                x = x + kappa * (mu - x) + sigma * rng.normal()
                out[t] = x
            return out

        series = {
            "CPI_YOY": ar1(2.0, 0.05, 0.25, 2.4),
            "GDP_YOY": ar1(2.5, 0.08, 0.30, 2.2),
            "UNEMPLOYMENT": ar1(5.0, 0.06, 0.15, 4.5),
            "FEDFUNDS": np.clip(ar1(0.5, 0.05, 0.20, 2.5), 0.0, 6.0),
            "YIELD_10Y": np.clip(ar1(2.5, 0.04, 0.18, 3.0), 0.3, 6.0),
            "TERM_SPREAD": ar1(1.2, 0.10, 0.12, 1.0),
            "VIX": np.clip(ar1(16.0, 0.20, 1.6, 18.0), 9.0, 80.0),
            "CREDIT_SPREAD": np.clip(ar1(1.2, 0.12, 0.10, 1.3), 0.3, 8.0),
            "USD_INDEX": ar1(98.0, 0.03, 0.9, 100.0),
        }
        rows = [
            {"date": d, "series_id": sid, "value": float(v)}
            for sid, values in series.items()
            for d, v in zip(dates, values)
        ]
        df = pd.DataFrame(rows)[MACRO_COLUMNS]
        log.info(f"Simulated macro series: {df['series_id'].nunique()} series | {len(df):,} rows")
        return df

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "provenance": PROVENANCE,
            "synthetic": True,
            "embedded_premia": EMBEDDED_PREMIA,
            "exposure_scale": EXPOSURE_SCALE,
            "spec": dict(self.spec.__dict__),
            "capabilities": {
                "prices": True,
                "fundamentals": True,
                "constituents": True,
                "macro": True,
                "delistings": True,
            },
        }


__all__ = ["SampleDataProvider", "SampleSpec", "EMBEDDED_PREMIA", "PROVENANCE"]
