# Research Copilot — `alphaforge.agents`

The copilot turns real tool outputs into a plain-language research briefing. It
is a research assistant, not an authority: its value is in surfacing what the
numbers say, not in saying what a manager wants to hear.

```python
from alphaforge.agents.copilot import ResearchCopilot, CopilotConfig

copilot = ResearchCopilot(CopilotConfig.from_dict(cfg))
briefing = copilot.analyze(state)   # state = the same ResearchState the API holds
print(briefing.to_text())
```

## Tool layer — the only thing the copilot can read

`alphaforge.agents.tools` wraps the real outputs of every upstream stage and
returns plain Python objects, **never prose**. The copilot reasons over the
returned numbers; it never invents them. If a stage was not run, the tool returns
`None` and the copilot says so. Tools include:

* `factor_summary_table` — top factors ranked by ICIR.
* `model_evaluation` — out-of-sample IC, ICIR, turnover, long-short spread.
* `backtest_metrics` / `backtest_diagnostics` — headline stats and run diagnostics.
* `risk_decomposition` — factor/specific risk split and top exposures.
* `attribution_summary` — Brinson (sectors) + factor attribution.
* `regime` / `stress` / `quality` — regime counts, scenario P&L, data-quality flags.

## Two modes

* **`none` (default) — deterministic analyst.** Fixed rules turn the tool
  results into a briefing. Every sentence is grounded in a number it actually
  received. Nothing is hallucinated.
* **`openai` / `anthropic`.** The same rules run first to produce a structured
  brief, which is handed to the LLM for prose. If the credential is missing or
  the call fails, the **deterministic brief is returned verbatim** — no
  fabrication on the fallback path.

## The rules

`RULES` is an explicit, auditable list (not a prompt). Examples: *Sharpe > 1.0*
→ "strong risk-adjusted return"; *0.5–1.0* → "moderate"; *< 0.5* → "weak — review
alpha"; *avg_turnover > 1.0* → "turnover is high; transaction costs dominate";
*cost drag > 2% of NAV* → flagged; *max drawdown < −30%* → "check risk budget";
*Rank-IC > 0.01 / 0–0.01 / ≤ 0* → economically meaningful / small / useless.

Each rule is wrapped so a failing rule can never break the brief.

## The briefing

`Briefing` carries four sections:

* **headline** — e.g. `Strategy CAGR +0.49%, Sharpe 0.10 (model Rank-IC +0.0447).`
* **findings** — what the numbers support (IC, Brinson allocation/selection,
  modal regime and its return split, worst stress scenario).
* **warnings** — cost bleed, deep drawdown, stress losses beyond −10%.
* **reproducibility checks** — walk-forward CV used (no in-sample IC), signals
  executed `lag` sessions after the signal date (look-ahead guarded), survivorship
  flagged in the data report.

## Default-mode reproducibility

The default mode has no free-form LLM call. Its findings are functions of tool
outputs and fixed rules. An optional LLM can change the prose; its response is
not guaranteed to preserve every number or interpretation and needs review.
A failed LLM call falls back to the deterministic briefing.
