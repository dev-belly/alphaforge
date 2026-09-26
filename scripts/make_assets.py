"""Generate the README sample-output figures from a real pipeline run.

Headless (matplotlib Agg). Runs the *shipped synthetic sample* end-to-end and
renders a few canonical charts into ``assets/`` so the repo has real visuals
without needing a browser screenshot. Deterministic given the project seed.

    python scripts/make_assets.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alphaforge.pipeline import ResearchPipeline
from alphaforge.utils.config import Config, load_yaml

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = "#1f4e79"
TEAL = "#2a9d8f"
GREY = "#999999"
RED = "#b22222"


def render_assets(state) -> None:
    """Render the README figures from the same state used by the sample report."""
    if state.config.get("data", {}).get("provider") != "sample":
        raise ValueError("README figures must use the synthetic sample provider")
    bt = state.backtest

    # ---- equity curve + drawdown -----------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    eq = bt.equity.dropna()
    ax1.plot(eq.index, eq.values, color=NAVY, lw=1.4, label="Strategy (net)")
    if bt.gross_equity is not None:
        ge = bt.gross_equity.dropna()
        ax1.plot(ge.index, ge.values, color=TEAL, lw=1.0, ls="--", label="Strategy (gross)")
    if bt.benchmark is not None:
        bm = (1 + bt.benchmark.reindex(eq.index)).cumprod() * bt.config["initial_capital"]
        ax1.plot(bm.index, bm.values, color=GREY, lw=1.0, ls=":", label="Synthetic benchmark")
    ax1.set_title(f"AlphaForge - synthetic sample ({eq.index[0].year}-{eq.index[-1].year})")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    dd = eq / eq.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.4)
    ax2.set_title("Drawdown", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "equity_curve.png", dpi=130)
    plt.close(fig)

    # ---- top factors by Rank-IC ------------------------------------------
    fs = state.factor_summary
    if fs is not None and not fs.empty and "factor" in fs.columns:
        sort_col = "rank_ic_mean" if "rank_ic_mean" in fs.columns else "ic_mean"
        top = fs.sort_values(sort_col, ascending=False).head(15).copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = top[sort_col].to_numpy(dtype=float)
        colors = [NAVY if v >= 0 else RED for v in vals]
        ax.barh(top["factor"][::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"Top factors by {sort_col} (of {len(fs)} evaluated)")
        ax.set_xlabel(sort_col)
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(ASSETS / "factor_ic.png", dpi=130)
        plt.close(fig)

    # ---- annualized return by regime -------------------------------------
    rs = state.diagnostics.get("regime_stats", {})
    if rs:
        labels = list(rs.keys())
        ann = [float(rs[lab].get("ann_return", np.nan)) for lab in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [TEAL if v >= 0 else RED for v in ann]
        ax.bar(labels, ann, color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("Annualized return by market regime")
        ax.set_ylabel("ann. return")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(ASSETS / "regime_returns.png", dpi=130)
        plt.close(fig)

    written = sorted(p.name for p in ASSETS.glob("*.png"))
    print("assets written:", written)


if __name__ == "__main__":
    config = Config(raw=load_yaml(ASSETS.parent / "configs/default.yaml"))
    if config.get("data.provider") != "sample":
        raise ValueError("README figures must use the synthetic sample provider")
    render_assets(ResearchPipeline(config).run())
