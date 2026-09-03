"""Self-contained HTML research report.

A report is one HTML file: base64-embedded figures, rendered tables, and a
plain-language summary.  It is generated from the *real* outputs of every
earlier stage (factors, model, backtest, attribution, risk) so it can never
present a number the engine did not actually produce.  There is no prose
generation here - the interpretation is the job of the research copilot
(:mod:`alphaforge.agents`).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.reporting import charts
from alphaforge.utils.logging import get_logger

log = get_logger("reporting.report")

PCT_KEYS = {
    "total_return",
    "cagr",
    "ann_vol",
    "ann_downside_dev",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "avg_turnover",
    "hit_rate",
    "positive_period_share",
    "benchmark_return",
    "benchmark_cagr",
    "benchmark_vol",
    "active_return",
    "alpha_ann",
    "tracking_error",
    "information_ratio",
    "up_capture",
    "down_capture",
    "treynor",
    "cost_drag_ann",
}


@dataclass
class ReportInputs:
    """Whatever the pipeline produced, gathered for rendering."""

    title: str = "AlphaForge Research Report"
    config: dict = field(default_factory=dict)
    factor_summary: pd.DataFrame | None = None
    model_summary: dict | None = None
    backtest: Any | None = None
    risk_decomposition: Any | None = None
    brinson: Any | None = None
    factor_attribution: Any | None = None
    regime: Any | None = None
    regime_stats: dict | None = None
    stress: Any | None = None
    notes: list[str] = field(default_factory=list)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 1e-6:
            return "0.0000"
        if abs(value) >= 1e6 or abs(value) < 1e-3:
            return f"{value:.4g}"
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return _fmt(value)


def _table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "<p><i>No data.</i></p>"
    cols = list(df.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                txt = floatfmt.format(v)
            else:
                txt = html.escape(str(v))
            cells.append(f"<td>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table class='data'><thead><tr>{head}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _kv_table(metrics: dict) -> str:
    rows = []
    for k, v in metrics.items():
        val = _pct(v) if k in PCT_KEYS else _fmt(v)
        rows.append(f"<tr><td class='k'>{html.escape(str(k))}</td><td class='v'>{val}</td></tr>")
    return f"<table class='kv'>{''.join(rows)}</table>"


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" />' if b64 else ""


def build_html(inputs: ReportInputs) -> str:
    """Render the full report to an HTML string."""
    bt = inputs.backtest
    equity_b64 = drawdown_b64 = monthly_b64 = rc_b64 = brinson_b64 = quantile_b64 = ""
    if bt is not None:
        equity_b64 = _img(charts.equity_curve(bt.equity, getattr(bt, "benchmark", None)))
        drawdown_b64 = _img(charts.drawdown(bt.equity))
        monthly = _safe_monthly(bt.returns)
        monthly_b64 = _img(charts.monthly_heatmap(monthly))
        if bt.metrics:
            rc_cov = _cov_from_risk(inputs)
            if rc_cov is not None:
                rc_b64 = _img(charts.risk_contribution(_avg_weights(bt.weights), rc_cov))

    if inputs.model_summary is not None:
        quantile_b64 = _img(charts.quantile_bar(_quantile_from_model(inputs)))

    if inputs.brinson is not None:
        brinson_b64 = _img(charts.brinson_bars(inputs.brinson))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    notes_html = (
        "".join(f"<li>{html.escape(n)}</li>" for n in inputs.notes) or "<li><i>None.</i></li>"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>{html.escape(inputs.title)}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0; color: #222; }}
.wrap {{ max-width: 960px; margin: 0 auto; padding: 28px 22px 60px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; margin: 26px 0 8px; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; }}
.sub {{ color: #777; font-size: 12px; margin-bottom: 4px; }}
img {{ width: 100%; border: 1px solid #eee; border-radius: 4px; margin: 6px 0; }}
table.data {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 6px 0; }}
table.data th, table.data td {{ border: 1px solid #e3e3e3; padding: 4px 7px; text-align: right; }}
table.data th {{ background: #1f4e79; color: #fff; }}
table.data tr:nth-child(even) td {{ background: #f7f9fc; }}
table.kv {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
table.kv td {{ border: 1px solid #eee; padding: 4px 9px; }}
table.kv td.k {{ background: #f3f6fb; width: 45%; font-weight: 600; }}
ul.notes {{ font-size: 12.5px; color: #444; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.foot {{ color: #999; font-size: 11px; margin-top: 30px; }}
</style></head>
<body><div class="wrap">
<h1>{html.escape(inputs.title)}</h1>
<div class="sub">Generated {now}</div>

<h2>Performance</h2>
{equity_b64}
{drawdown_b64}
{_kv_table(bt.metrics) if bt is not None and bt.metrics else "<p><i>No backtest run.</i></p>"}
{("<h2>Monthly Returns</h2>" + monthly_b64) if monthly_b64 else ""}

<h2>Factor Research</h2>
{_table(inputs.factor_summary)}
{quantile_b64}

<div class="grid">
{"<div><h2>Risk Decomposition</h2>" + rc_b64 + _table(inputs.risk_decomposition) + "</div>" if inputs.risk_decomposition is not None and not _is_empty(inputs.risk_decomposition) else ""}
{"<div><h2>Attribution (Factor Bets)</h2>" + _table(_attr_table(inputs.factor_attribution)) + "</div>" if inputs.factor_attribution is not None else ""}
</div>

    {("<h2>Brinson Attribution</h2>" + brinson_b64 + _table(inputs.brinson.by_sector)) if inputs.brinson is not None else ""}

{_regime_section(inputs.regime, inputs.regime_stats)}

{_stress_section(inputs.stress)}

<h2>Notes &amp; Caveats</h2>
<ul class="notes">{notes_html}</ul>

<div class="foot">AlphaForge - institutional quant research &amp; portfolio engineering. Figures are
research outputs, not investment advice.</div>
</div></body></html>"""


def _is_empty(df) -> bool:
    return df is None or (isinstance(df, pd.DataFrame) and df.empty)


def _safe_monthly(returns: pd.Series):
    try:
        from alphaforge.backtest.metrics import monthly_returns

        return monthly_returns(returns)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _avg_weights(weights: pd.DataFrame) -> pd.Series:
    return weights.abs().mean(axis=0)


def _quantile_from_model(inputs: ReportInputs):
    try:
        return inputs.model_summary["quantile_returns"]
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _cov_from_risk(inputs: ReportInputs):
    try:
        return inputs.risk_decomposition.attrs.get("covariance")
    except Exception:  # noqa: BLE001
        return None


def _regime_section(regime: Any, stats: dict | None) -> str:
    if regime is None:
        return ""
    if not isinstance(regime, pd.Series) or regime.dropna().empty:
        return ""
    counts = {str(k): int(v) for k, v in regime.value_counts(dropna=True).items()}
    total = sum(counts.values()) or 1
    bar = "".join(
        f"<div style='margin:2px 0'><span style='display:inline-block;width:90px'>{html.escape(k)}</span>"
        f"<span style='display:inline-block;height:12px;background:#1f4e79;border-radius:2px;width:{max(2, int(v / total * 200))}px'></span>"
        f" {v} ({v / total:.0%})</div>"
        for k, v in counts.items()
    )
    table = ""
    if stats:
        rows = []
        for lab, s in stats.items():
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(x))}</td>"
                    for x in [
                        lab,
                        s.get("n_days"),
                        _pct(s.get("ann_return")),
                        _pct(s.get("ann_vol")),
                        _fmt(s.get("sharpe")),
                    ]
                )
                + "</tr>"
            )
        table = (
            "<table class='data'><thead><tr><th>regime</th><th>n_days</th><th>ann_return</th>"
            "<th>ann_vol</th><th>sharpe</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    return (
        "<h2>Market Regime</h2>"
        "<p class='sub'>Per-day regime from benchmark trend &times; volatility (no future data).</p>"
        f"<div>{bar}</div>{table}"
    )


def _stress_section(stress: Any) -> str:
    if not stress:
        return ""
    rows = []
    for nm, res in stress.items():
        worst = res.worst_holdings(3)
        worst_txt = ", ".join(f"{w['symbol']} {w['contribution']:+.2%}" for w in worst)
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(x))}</td>"
                for x in [nm, _pct(res.pnl_pct), worst_txt or "-"]
            )
            + "</tr>"
        )
    return (
        "<h2>Stress Testing</h2>"
        "<p class='sub'>Scenario P&amp;L from factor shocks on the representative book "
        "(time-mean of absolute weights). Deterministic, no Monte Carlo.</p>"
        "<table class='data'><thead><tr><th>scenario</th><th>pnl</th>"
        "<th>worst holdings</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _attr_table(fa) -> pd.DataFrame:
    if fa is None:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "factor": fa.betas.index,
            "beta": fa.betas.to_numpy(dtype=float),
            "attributed_return": fa.attributed_return.reindex(fa.betas.index).to_numpy(dtype=float),
            "t_stat": fa.t_stats.reindex(fa.betas.index).to_numpy(dtype=float),
        }
    )
    return df


def write_report(inputs: ReportInputs, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(inputs), encoding="utf-8")
    log.info(f"Report written to {path}")
    return path


__all__ = ["ReportInputs", "build_html", "write_report"]
