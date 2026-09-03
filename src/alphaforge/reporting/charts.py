"""Static figures for the research report.

Every chart is rendered to PNG bytes (base64) so the HTML report is fully
self-contained - it can be emailed or opened offline with no asset server.  The
Agg backend is used because this module runs headless in CI and on the API box.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alphaforge.utils.logging import get_logger  # noqa: E402

log = get_logger("reporting.charts")

BLUE = "#1f4e79"
RED = "#c0392b"
GREEN = "#1e8449"
GREY = "#7f8c8d"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")


def equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    gross: pd.Series | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if gross is not None and not gross.dropna().empty:
        g = gross.reindex(equity.index)
        ax.plot(
            g.index,
            g.to_numpy(),
            color=GREEN,
            lw=1.0,
            ls="--",
            label="Gross (pre-cost)",
            alpha=0.85,
        )
    ax.plot(equity.index, equity.to_numpy(), color=BLUE, lw=1.4, label="Net (after-cost)")
    if benchmark is not None and not benchmark.dropna().empty:
        b = benchmark.reindex(equity.index).dropna()
        ax.plot(b.index, b.to_numpy(), color=GREY, lw=1.0, label="Benchmark", alpha=0.8)
    ax.set_title("Equity Curve (Net vs Gross)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _fig_to_b64(fig)


def drawdown(equity: pd.Series) -> str:
    curve = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.fill_between(curve.index, curve.to_numpy() * 100, 0, color=RED, alpha=0.35)
    ax.plot(curve.index, curve.to_numpy() * 100, color=RED, lw=0.8)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("%")
    _style(ax)
    return _fig_to_b64(fig)


def monthly_heatmap(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, max(1.6, 0.4 * len(table))))
    data = table.to_numpy(dtype=float) * 100
    vmax = float(np.nanmax(np.abs(data))) or 1.0
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, fontsize=7)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index, fontsize=7)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if abs(val) < vmax * 0.6 else "white",
                )
    ax.set_title("Monthly Returns (%)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    return _fig_to_b64(fig)


def ic_series(ic: pd.Series) -> str:
    if ic.dropna().empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.plot(ic.index, ic.cumsum().to_numpy(), color=GREEN, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("Cumulative Rank-IC")
    _style(ax)
    return _fig_to_b64(fig)


def quantile_bar(quantile_returns: pd.DataFrame) -> str:
    if quantile_returns.empty:
        return ""
    if "mean" in quantile_returns.columns:
        means = quantile_returns["mean"]
    else:
        means = quantile_returns.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    xs = np.arange(len(means))
    ax.bar(xs, means.to_numpy() * 100, color=[RED if v < 0 else GREEN for v in means.to_numpy()])
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i + 1) for i in xs], fontsize=8)
    ax.set_title("Mean Return by Signal Quintile (%)")
    ax.set_xlabel("Quintile (1 = lowest signal)")
    _style(ax)
    return _fig_to_b64(fig)


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> str:
    if weights.dropna().empty:
        return ""
    cols = [c for c in weights.index if abs(weights[c]) > 1e-6]
    if not cols:
        return ""
    w = weights[cols].to_numpy(dtype=float)
    c = cov.reindex(index=cols, columns=cols).to_numpy(dtype=float)
    port_var = float(w @ c @ w) or 1.0
    rc = w * (c @ w)
    contrib = rc / port_var * 100
    order = np.argsort(-contrib)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.bar(np.arange(len(order)), contrib[order], color=BLUE)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([cols[i] for i in order], rotation=60, fontsize=6, ha="right")
    ax.set_title("Risk Contribution by Holding (%)")
    _style(ax)
    return _fig_to_b64(fig)


def brinson_bars(result) -> str:
    if result.by_sector.empty:
        return ""
    df = result.by_sector.set_index("sector")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.bar(x - 0.2, df["allocation"].to_numpy() * 100, 0.4, label="Allocation", color=BLUE)
    ax.bar(x + 0.2, df["selection"].to_numpy() * 100, 0.4, label="Selection", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=7)
    ax.set_title("Brinson Attribution by Sector (%, per period)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _fig_to_b64(fig)


__all__ = [
    "equity_curve",
    "drawdown",
    "monthly_heatmap",
    "ic_series",
    "quantile_bar",
    "risk_contribution",
    "brinson_bars",
]
