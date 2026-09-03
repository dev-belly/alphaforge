"""Brinson-Fachler performance attribution.

Answers the only question an allocator actually asks after a period: *how much
of the active return came from being in the right sectors versus picking the
right names inside them?*  Brinson-Fachler (1986) splits active return into:

    active      = allocation + selection + interaction

    allocation   = Σ_s (w_ps - w_bs) · (r_bs - r_b)
    selection    = Σ_s w_ps · (r_ps - r_bs)
    interaction  = Σ_s (w_ps - w_bs) · (r_ps - r_bs)

where ``w`` is weight, ``r`` is return and the ``p``/``b`` subscripts are the
portfolio / benchmark.  ``r_bs`` and ``r_ps`` are the *within-sector* returns,
so a sector that the portfolio over-weights but where every name lags the
benchmark shows up as selection pain, not allocation gain.

All inputs are cross-sectional per date; aggregation to a total is a weighted
average of per-period effects, never a naive sum of daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("attribution.brinson")


@dataclass
class BrinsonResult:
    """Per-period and per-sector attribution of active return."""

    total_active: float
    allocation: float
    selection: float
    interaction: float
    by_sector: pd.DataFrame
    by_period: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "total_active": float(self.total_active),
            "allocation": float(self.allocation),
            "selection": float(self.selection),
            "interaction": float(self.interaction),
            "explained_share": float(
                (self.allocation + self.selection + self.interaction) / self.total_active
                if self.total_active
                else float("nan")
            ),
        }


def _period_returns(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """Portfolio return per date from (dates x symbols) weights and returns."""
    w = weights.reindex(index=asset_returns.index, columns=asset_returns.columns).fillna(0.0)
    r = asset_returns.reindex(index=asset_returns.index, columns=asset_returns.columns).fillna(0.0)
    return (w * r).sum(axis=1)


def brinson_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    sectors: pd.Series | pd.DataFrame,
) -> BrinsonResult:
    """Decompose active return for every date with a sector map.

    Parameters
    ----------
    portfolio_weights, benchmark_weights:
        (dates x symbols), weights *before* each date's return.  They need not
        sum to one (a residual cash weight is treated as a sector of its own).
    asset_returns:
        (dates x symbols) one-period returns.
    sectors:
        ``symbol -> sector`` (Series) or (dates x symbols) assignment.  Names
        with no sector land in ``UNCLASSIFIED``.

    Notes
    -----
    The benchmark return per date is used as the market return ``r_b``; if the
    benchmark is equal-weighted (default of the risk model), the attribution is
    still valid - it just reflects deviations from that baseline.
    """
    dates = asset_returns.index
    symbols = asset_returns.columns
    wp = portfolio_weights.reindex(index=dates, columns=symbols).fillna(0.0)
    wb = benchmark_weights.reindex(index=dates, columns=symbols).fillna(0.0)
    r = asset_returns.reindex(index=dates, columns=symbols).fillna(0.0)

    if isinstance(sectors, pd.Series):
        sector_map = sectors.reindex(symbols).astype("object").fillna("UNCLASSIFIED")
    else:
        sector_map = sectors.reindex(index=dates, columns=symbols).fillna("UNCLASSIFIED")
    sector_labels = sorted(set(np.asarray(sector_map).ravel()) - {"UNCLASSIFIED"}) or [
        "UNCLASSIFIED"
    ]

    rp = _period_returns(wp, r)
    rb = _period_returns(wb, r)

    period_rows: list[dict] = []

    for t in dates:
        wp_t = wp.loc[t]
        wb_t = wb.loc[t]
        r_t = r.loc[t]
        if isinstance(sector_map, pd.DataFrame):
            sect_t = sector_map.loc[t].reindex(symbols).fillna("UNCLASSIFIED")
        else:
            sect_t = sector_map

        alloc = sel = inter = 0.0
        for s in sector_labels:
            mask = sect_t.to_numpy() == s
            wps = float(wp_t[mask].sum())
            wbs = float(wb_t[mask].sum())
            if wps == 0 and wbs == 0:
                continue
            rps = float(wp_t[mask] @ r_t[mask]) / wps if wps > 0 else 0.0
            rbs = float(wb_t[mask] @ r_t[mask]) / wbs if wbs > 0 else 0.0
            alloc += (wps - wbs) * (rbs - rb.loc[t])
            sel += wps * (rps - rbs)
            inter += (wps - wbs) * (rps - rbs)
        period_rows.append(
            {
                "date": t,
                "active": rp.loc[t] - rb.loc[t],
                "allocation": alloc,
                "selection": sel,
                "interaction": inter,
            }
        )

    by_period = pd.DataFrame(period_rows).set_index("date")
    # Aggregate to total by averaging per-period effects (periodic, not summed).
    total_alloc = float(by_period["allocation"].mean())
    total_sel = float(by_period["selection"].mean())
    total_inter = float(by_period["interaction"].mean())
    total_active = float((rp - rb).mean())

    # Per-sector table: attribute each period effect to its sector.
    sector_rows = []
    for t in dates:
        wp_t = wp.loc[t]
        wb_t = wb.loc[t]
        r_t = r.loc[t]
        sect_t = sector_map.loc[t] if isinstance(sector_map, pd.DataFrame) else sector_map
        for s in sector_labels:
            mask = sect_t.to_numpy() == s
            wps = float(wp_t[mask].sum())
            wbs = float(wb_t[mask].sum())
            if wps == 0 and wbs == 0:
                continue
            rps = float(wp_t[mask] @ r_t[mask]) / wps if wps > 0 else 0.0
            rbs = float(wb_t[mask] @ r_t[mask]) / wbs if wbs > 0 else 0.0
            sector_rows.append(
                {
                    "sector": s,
                    "portfolio_weight": wps,
                    "benchmark_weight": wbs,
                    "allocation": (wps - wbs) * (rbs - rb.loc[t]),
                    "selection": wps * (rps - rbs),
                    "interaction": (wps - wbs) * (rps - rbs),
                    "active": wps * rps - wbs * rbs,
                }
            )
    by_sector = pd.DataFrame(sector_rows)
    if not by_sector.empty:
        by_sector = (
            by_sector.groupby("sector", as_index=False)
            .mean()
            .sort_values("active", ascending=False)
        )
        by_sector = by_sector.reset_index(drop=True)

    log.info(
        f"Brinson: active={total_active:+.4%} | alloc={total_alloc:+.4%} | "
        f"sel={total_sel:+.4%} | inter={total_inter:+.4%}"
    )
    return BrinsonResult(
        total_active=total_active,
        allocation=total_alloc,
        selection=total_sel,
        interaction=total_inter,
        by_sector=by_sector,
        by_period=by_period,
    )


__all__ = ["BrinsonResult", "brinson_attribution"]
