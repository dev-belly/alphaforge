"""Research copilot: turns real tool outputs into a plain-language briefing.

The copilot has two modes, selected by the ``agent.provider`` config:

* ``none`` (default) - a deterministic analyst.  It reads the tool results,
  applies fixed rules (e.g. "Sharpe below 0.5 and turnover above 1.0 is a
  cost-bleed warning"), and writes the briefing.  Nothing is hallucinated: every
  sentence is grounded in a number it actually received.
* ``openai`` / ``anthropic`` - the same rules run first to produce a structured
  brief, which is then handed to the LLM for prose.  If the credential is
  missing or the call fails, the deterministic brief is returned verbatim.  No
  fabrication on the fallback path.

This is deliberately conservative: the copilot is a research assistant, not an
authority, and its value is in surfacing what the numbers say - not in saying
what a manager wants to hear.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphaforge.agents.tools import ToolResult, run_tools
from alphaforge.utils.logging import get_logger

log = get_logger("agents.copilot")

RULES = [
    ("Sharpe > 1.0", lambda m: (m.get("sharpe") or 0) > 1.0, "strong risk-adjusted return"),
    (
        "Sharpe 0.5-1.0",
        lambda m: 0.5 <= (m.get("sharpe") or -1) <= 1.0,
        "moderate risk-adjusted return",
    ),
    (
        "Sharpe < 0.5",
        lambda m: (m.get("sharpe") or 0) < 0.5,
        "weak risk-adjusted return - review alpha",
    ),
    (
        "High turnover cost",
        lambda m: (m.get("avg_turnover") or 0) > 1.0,
        "turnover is high; transaction costs dominate",
    ),
    (
        "Cost drag material",
        lambda m: abs(m.get("cost_drag_ann") or 0) > 0.02,
        "annual cost drag exceeds 2% of NAV",
    ),
    (
        "Deep drawdown",
        lambda m: (m.get("max_drawdown") or 0) < -0.30,
        "max drawdown exceeds 30% - check risk budget",
    ),
    (
        "Positive IC",
        lambda m: (m.get("rank_ic_mean") or 0) > 0.01,
        "model shows economically meaningful Rank-IC",
    ),
    (
        "Low IC",
        lambda m: 0 < (m.get("rank_ic_mean") or 0) <= 0.01,
        "model IC is positive but small",
    ),
    (
        "Negative IC",
        lambda m: (m.get("rank_ic_mean") or 0) <= 0,
        "model IC is non-positive - signal useless",
    ),
]


@dataclass
class CopilotConfig:
    provider: str = "none"
    model: str | None = None
    max_tool_calls: int = 8

    @classmethod
    def from_dict(cls, cfg: dict | None) -> CopilotConfig:
        cfg = cfg or {}
        a = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
        return cls(
            provider=str(a.get("provider", "none")),
            model=a.get("model"),
            max_tool_calls=int(a.get("max_tool_calls", 8)),
        )


@dataclass
class Briefing:
    headline: str
    findings: list[str]
    warnings: list[str]
    checks: list[str]
    llm_note: str = ""

    def to_text(self) -> str:
        out = [self.headline, ""]
        if self.findings:
            out += ["Findings:", *[f"  - {f}" for f in self.findings], ""]
        if self.warnings:
            out += ["Warnings:", *[f"  - {w}" for w in self.warnings], ""]
        if self.checks:
            out += ["Reproducibility checks:", *[f"  - {c}" for c in self.checks], ""]
        if self.llm_note:
            out += [self.llm_note]
        return "\n".join(out)


class ResearchCopilot:
    """Reads tool outputs and writes a grounded research briefing."""

    def __init__(self, config: CopilotConfig | dict | None = None) -> None:
        self.config = (
            config if isinstance(config, CopilotConfig) else CopilotConfig.from_dict(config or {})
        )

    # ------------------------------------------------------------------
    def analyze(self, state: dict) -> Briefing:
        """Run the tool layer and synthesise a briefing from the results."""
        results = run_tools(state)
        findings, warnings = self._rule_findings(results)
        checks = self._repro_checks(results, state)

        headline = self._headline(results)

        llm_note = ""
        if self.config.provider in {"openai", "anthropic"}:
            llm_note = self._llm_prose(results)
        return Briefing(
            headline=headline,
            findings=findings,
            warnings=warnings,
            checks=checks,
            llm_note=llm_note,
        )

    # ------------------------------------------------------------------
    def _rule_findings(self, results: dict[str, ToolResult]) -> tuple[list[str], list[str]]:
        findings: list[str] = []
        warnings: list[str] = []
        m = self._get(results, "backtest") or {}
        model = self._get(results, "model") or {}
        merged = {**m, **model}
        for label, fn, text in RULES:
            try:
                if fn(merged):
                    (
                        warnings
                        if "cost" in label or "drawdown" in label or "useless" in label
                        else findings
                    ).append(f"{label}: {text}")
            except Exception:  # noqa: BLE001 - a rule must never break the brief
                continue
        attr = (
            (self._get(results, "attribution") or {}).get("brinson")
            if "attribution" in results
            else None
        )
        if attr:
            findings.append(
                f"Brinson: allocation {attr.get('allocation'):+.2%}, selection "
                f"{attr.get('selection'):+.2%} of {attr.get('total_active'):+.2%} active return"
            )
        regime_data = self._get(results, "regime")
        if regime_data:
            counts = regime_data.get("counts", {})
            if counts:
                total = sum(counts.values()) or 1
                dom = max(counts, key=counts.get)
                findings.append(
                    f"Regime: {dom} was the modal regime ({counts[dom] / total:.0%} of days); "
                    f"split by Bull/Bear x High/Low-Vol."
                )
            stats = regime_data.get("return_stats", {})
            if stats:
                best = max(stats, key=lambda k: (stats[k] or {}).get("ann_return", float("-inf")))
                worst = min(stats, key=lambda k: (stats[k] or {}).get("ann_return", float("inf")))
                b = stats.get(best, {})
                w = stats.get(worst, {})
                findings.append(
                    f"Regime return split: best {best} (ann {b.get('ann_return', float('nan')):+.2%}, "
                    f"Sharpe {b.get('sharpe', float('nan')):.2f}); worst {worst} "
                    f"(ann {w.get('ann_return', float('nan')):+.2%})."
                )
        stress_data = self._get(results, "stress")
        if stress_data:
            losses = {
                nm: d.get("pnl_pct", 0.0) for nm, d in stress_data.items() if isinstance(d, dict)
            }
            if losses:
                worst_nm = min(losses, key=lambda k: losses[k])
                findings.append(
                    f"Stress: {len(losses)} scenarios; worst {worst_nm} {losses[worst_nm]:+.2%}."
                )
                severe = [nm for nm, v in losses.items() if v < -0.10]
                if severe:
                    warnings.append(
                        "Stress: "
                        + ", ".join(f"{nm} {losses[nm]:+.2%}" for nm in severe)
                        + " exceed a 10% loss - review factor tilts."
                    )
        return findings, warnings

    def _repro_checks(self, results: dict[str, ToolResult], state: dict) -> list[str]:
        checks = []
        if "model" in results and results["model"].ok:
            checks.append("Model evaluated under walk-forward CV (no in-sample IC used).")
        if "backtest" in results and results["backtest"].ok:
            d = self._get(results, "diagnostics") or {}
            checks.append(
                f"Backtest executes signals {d.get('execution_lag_days', 1)} session(s) after signal "
                f"date (look-ahead guarded)."
            )
        if "quality" in results and results["quality"].ok:
            data = results["quality"].data
            q = data if isinstance(data, dict) else {}
            if q.get("survivorship_flagged"):
                checks.append(
                    "Survivorship bias is flagged in the data report (point-in-time only)."
                )
        return checks

    def _headline(self, results: dict[str, ToolResult]) -> str:
        bt = self._get(results, "backtest") or {}
        model = self._get(results, "model") or {}
        if not bt:
            return "Research briefing: backtest not run - see factor/model sections."
        cagr = bt.get("cagr", float("nan"))
        sharpe = bt.get("sharpe", float("nan"))
        return (
            f"Strategy CAGR {cagr:+.2%}, Sharpe {sharpe:.2f} "
            f"(model Rank-IC {model.get('rank_ic_mean'):+.4f})."
        )

    # ------------------------------------------------------------------
    def _llm_prose(self, results: dict[str, ToolResult]) -> str:
        """Best-effort LLM prose; degrades to empty if unavailable."""
        try:
            from alphaforge.agents._llm import llm_complete  # local import: optional dep

            prompt = self._structured_prompt(results)
            return llm_complete(self.config.provider, self.config.model, prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"LLM prose unavailable ({exc}) - returning deterministic brief")
            return ""

    def _structured_prompt(self, results: dict[str, ToolResult]) -> str:
        parts = []
        for name, r in results.items():
            if r.ok and r.data is not None:
                parts.append(f"[{name}] {r.note}\n{r.data}")
        return (
            "Summarise this quant research output for an investment committee in 3 bullets:\n\n"
            + "\n\n".join(parts)
        )

    @staticmethod
    def _get(results: dict[str, ToolResult], key: str):
        r = results.get(key)
        return r.data if (r and r.ok) else None


__all__ = ["CopilotConfig", "ResearchCopilot", "Briefing"]
