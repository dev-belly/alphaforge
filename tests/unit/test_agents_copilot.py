"""Offline tests for the research copilot.

The copilot's whole claim is that it does not fabricate: "every sentence is
grounded in a number it actually received". That makes the failure that matters
a *fabricated sentence*, and the way to write one is to confuse "the value is
zero" with "there is no value". ``m.get(key) or 0`` collapses the two, so a
rule comparing against zero fires on absent data.

Two rules did exactly that. With a completely empty metrics dict the copilot
reported "Sharpe < 0.5: weak risk-adjusted return - review alpha" and, into
*warnings*, "Negative IC: model IC is non-positive - signal useless" - two
conclusions about numbers it had never seen. Both now require the key to be
present, and ``test_no_rule_fires_on_an_empty_metrics_dict`` enforces that as a
blanket invariant, so a future rule added with the same ``or 0`` shape fails
here rather than in a briefing.

``_headline`` had the sibling defect: ``cagr`` and ``sharpe`` carried NaN
defaults but ``rank_ic_mean`` did not, so a model block that omitted the field
raised TypeError out of the f-string and took the entire briefing with it.

Everything is deterministic: no network, no LLM, no shared state.
"""

from __future__ import annotations

import sys
import types

import pytest

from alphaforge.agents.copilot import RULES, Briefing, CopilotConfig, ResearchCopilot
from alphaforge.agents.tools import ToolResult


def _result(name: str, data, ok: bool = True) -> ToolResult:
    return ToolResult(name=name, ok=ok, data=data, note=f"{name} note")


def _copilot(**kwargs) -> ResearchCopilot:
    return ResearchCopilot(CopilotConfig(**kwargs))


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def test_the_default_provider_is_the_deterministic_one() -> None:
    assert CopilotConfig().provider == "none"
    assert CopilotConfig().max_tool_calls == 8


def test_from_dict_reads_the_agent_block() -> None:
    got = CopilotConfig.from_dict(
        {"agent": {"provider": "openai", "model": "gpt-x", "max_tool_calls": 3}}
    )
    assert got.provider == "openai"
    assert got.model == "gpt-x"
    assert got.max_tool_calls == 3


@pytest.mark.parametrize("cfg", [None, {}])
def test_an_empty_config_uses_the_defaults(cfg) -> None:
    assert CopilotConfig.from_dict(cfg) == CopilotConfig()


def test_the_copilot_accepts_a_config_object_or_a_dict() -> None:
    assert _copilot(provider="none").config.provider == "none"
    assert ResearchCopilot({"agent": {"provider": "none"}}).config.provider == "none"
    assert ResearchCopilot().config.provider == "none"


# ----------------------------------------------------------------------
# The rules must not fire on absent data
# ----------------------------------------------------------------------
def test_no_rule_fires_on_an_empty_metrics_dict() -> None:
    """Regression: "Sharpe < 0.5" and "Negative IC" both fired on ``{}``.

    ``m.get(key) or 0`` makes a missing value indistinguishable from a real
    zero, and ``0 < 0.5`` and ``0 <= 0`` are both true.
    """
    fired = [label for label, fn, _ in RULES if fn({})]
    assert fired == [], f"rules fired without any data: {fired}"


@pytest.mark.parametrize("key", ["sharpe", "rank_ic_mean"])
def test_a_rule_stays_silent_when_its_input_is_missing(key: str) -> None:
    others = {"cagr": 0.08, "avg_turnover": 0.2, "max_drawdown": -0.1}
    assert not any(fn(others) for _, fn, _ in RULES), f"a rule fired without {key}"
    assert not any(fn({**others, key: None}) for _, fn, _ in RULES)


def test_the_rules_fire_on_real_values() -> None:
    strong = {"sharpe": 1.6, "rank_ic_mean": 0.03}
    fired = {label for label, fn, _ in RULES if fn(strong)}
    assert "Sharpe > 1.0" in fired
    assert "Positive IC" in fired
    assert "Sharpe < 0.5" not in fired

    weak = {"sharpe": 0.2, "rank_ic_mean": -0.01, "avg_turnover": 1.5, "max_drawdown": -0.4}
    fired = {label for label, fn, _ in RULES if fn(weak)}
    assert {"Sharpe < 0.5", "Negative IC", "High turnover cost", "Deep drawdown"} <= fired


def test_a_genuine_zero_sharpe_is_still_reported_as_weak() -> None:
    """The guard must distinguish absent from zero, not silence both."""
    fired = {label for label, fn, _ in RULES if fn({"sharpe": 0.0})}
    assert "Sharpe < 0.5" in fired


def test_a_zero_ic_is_still_reported_as_non_positive() -> None:
    fired = {label for label, fn, _ in RULES if fn({"rank_ic_mean": 0.0})}
    assert "Negative IC" in fired


def test_the_sharpe_bands_are_exhaustive_and_disjoint() -> None:
    """Every finite Sharpe must land in exactly one band."""
    labels = ["Sharpe > 1.0", "Sharpe 0.5-1.0", "Sharpe < 0.5"]
    for sharpe in (-0.5, 0.0, 0.49, 0.5, 0.75, 1.0, 1.01, 3.0):
        hits = [label for label, fn, _ in RULES if label in labels and fn({"sharpe": sharpe})]
        assert len(hits) == 1, f"Sharpe {sharpe} matched {hits}"


# ----------------------------------------------------------------------
# _rule_findings
# ----------------------------------------------------------------------
def test_findings_and_warnings_are_routed_by_label() -> None:
    cp = _copilot()
    results = {
        "backtest": _result(
            "backtest",
            {"sharpe": 0.2, "avg_turnover": 2.0, "max_drawdown": -0.5, "cost_drag_ann": 0.05},
        ),
        "model": _result("model", {"rank_ic_mean": 0.03}),
    }
    findings, warnings = cp._rule_findings(results)
    assert any("Positive IC" in f for f in findings)
    assert any("Sharpe < 0.5" in f for f in findings)
    assert any("High turnover cost" in w for w in warnings)
    assert any("Deep drawdown" in w for w in warnings)
    assert not any("Positive IC" in w for w in warnings)


def test_a_broken_rule_does_not_break_the_brief(monkeypatch) -> None:
    monkeypatch.setattr(
        "alphaforge.agents.copilot.RULES",
        [("Explodes", lambda m: 1 / 0, "never"), *RULES],
    )
    findings, warnings = _copilot()._rule_findings({})
    assert findings == [] and warnings == []


def test_a_missing_backtest_tool_produces_no_rule_findings() -> None:
    assert _copilot()._rule_findings({}) == ([], [])


def test_a_failed_tool_result_is_ignored() -> None:
    """``ok=False`` means the tool did not produce data, whatever is in ``data``."""
    results = {"backtest": _result("backtest", {"sharpe": 0.2}, ok=False)}
    assert _copilot()._rule_findings(results) == ([], [])


def test_the_brinson_finding_is_formatted_from_the_tool_output() -> None:
    results = {
        "attribution": _result(
            "attribution",
            {"brinson": {"allocation": 0.002, "selection": -0.001, "total_active": 0.001}},
        )
    }
    findings, _ = _copilot()._rule_findings(results)
    assert len(findings) == 1
    assert "+0.20%" in findings[0] and "-0.10%" in findings[0]


def test_a_regime_finding_names_the_modal_regime() -> None:
    results = {
        "regime": _result("regime", {"counts": {"Bull": 300, "Bear": 100}}),
    }
    findings, _ = _copilot()._rule_findings(results)
    assert any("Bull" in f and "75%" in f for f in findings)


def test_a_regime_finding_ranks_the_best_and_worst_splits() -> None:
    results = {
        "regime": _result(
            "regime",
            {
                "counts": {"Bull": 10, "Bear": 10},
                "return_stats": {
                    "Bull": {"ann_return": 0.2, "sharpe": 1.1},
                    "Bear": {"ann_return": -0.15, "sharpe": -0.4},
                },
            },
        )
    }
    findings, _ = _copilot()._rule_findings(results)
    joined = " ".join(findings)
    assert "best Bull" in joined and "worst Bear" in joined


def test_an_empty_regime_block_produces_nothing() -> None:
    assert _copilot()._rule_findings({"regime": _result("regime", {})}) == ([], [])


def test_a_severe_stress_scenario_becomes_a_warning() -> None:
    results = {
        "stress": _result("stress", {"2008": {"pnl_pct": -0.35}, "mild": {"pnl_pct": -0.02}})
    }
    findings, warnings = _copilot()._rule_findings(results)
    assert any("2008" in f for f in findings)
    assert any("2008" in w and "10%" in w for w in warnings)
    assert not any("mild" in w for w in warnings)


def test_a_mild_stress_block_produces_no_warning() -> None:
    results = {"stress": _result("stress", {"mild": {"pnl_pct": -0.02}})}
    _, warnings = _copilot()._rule_findings(results)
    assert warnings == []


def test_a_non_dict_stress_payload_is_ignored() -> None:
    assert _copilot()._rule_findings({"stress": _result("stress", ["not", "a", "dict"])}) == (
        [],
        [],
    )


# ----------------------------------------------------------------------
# _repro_checks
# ----------------------------------------------------------------------
def test_repro_checks_report_the_walk_forward_and_the_execution_lag() -> None:
    results = {
        "model": _result("model", {}),
        "backtest": _result("backtest", {}),
        "diagnostics": _result("diagnostics", {"execution_lag_days": 2}),
    }
    checks = _copilot()._repro_checks(results, {})
    assert any("walk-forward" in c for c in checks)
    assert any("2 session" in c for c in checks)


def test_repro_checks_flag_survivorship_when_reported() -> None:
    results = {"quality": _result("quality", {"survivorship_flagged": True})}
    checks = _copilot()._repro_checks(results, {})
    assert any("Survivorship" in c for c in checks)


def test_repro_checks_are_empty_without_any_tool_output() -> None:
    assert _copilot()._repro_checks({}, {}) == []


def test_repro_checks_ignore_a_failed_tool() -> None:
    results = {"model": _result("model", {}, ok=False)}
    assert _copilot()._repro_checks(results, {}) == []


# ----------------------------------------------------------------------
# _headline
# ----------------------------------------------------------------------
def test_the_headline_quotes_the_backtest_and_the_model() -> None:
    results = {
        "backtest": _result("backtest", {"cagr": 0.08, "sharpe": 0.9}),
        "model": _result("model", {"rank_ic_mean": 0.03}),
    }
    headline = _copilot()._headline(results)
    assert "+8.00%" in headline and "0.90" in headline and "+0.0300" in headline


def test_the_headline_says_so_when_there_is_no_backtest() -> None:
    assert "backtest not run" in _copilot()._headline({})


@pytest.mark.parametrize("payload", [{}, {"n_periods": 100}, {"rank_ic_mean": None}])
def test_the_headline_survives_a_model_block_without_an_ic(payload) -> None:
    """Regression: ``rank_ic_mean`` had no NaN default, unlike its two siblings.

    ``f"{None:+.4f}"`` raises TypeError, which is not caught anywhere on the way
    out of ``analyze`` - one absent field took the whole briefing with it.
    """
    results = {
        "backtest": _result("backtest", {"cagr": 0.05, "sharpe": 0.8}),
        "model": _result("model", payload),
    }
    headline = _copilot()._headline(results)
    assert "nan" in headline.lower()


@pytest.mark.parametrize("payload", [{"cagr": None}, {"cagr": 0.05, "sharpe": None}])
def test_the_headline_survives_a_backtest_without_cagr_or_sharpe(payload) -> None:
    """``None`` is not covered by ``dict.get``'s default - only a missing key is."""
    results = {
        "backtest": _result("backtest", payload),
        "model": _result("model", {"rank_ic_mean": 0.01}),
    }
    assert "nan" in _copilot()._headline(results).lower()


def test_an_empty_backtest_payload_reads_as_not_run() -> None:
    results = {"backtest": _result("backtest", {}), "model": _result("model", {})}
    assert "backtest not run" in _copilot()._headline(results)


# ----------------------------------------------------------------------
# Briefing
# ----------------------------------------------------------------------
def test_to_text_lays_out_every_section() -> None:
    briefing = Briefing(
        headline="Head",
        findings=["f1"],
        warnings=["w1"],
        checks=["c1"],
        llm_note="note",
    )
    text = briefing.to_text()
    for fragment in (
        "Head",
        "Findings:",
        "  - f1",
        "Warnings:",
        "  - w1",
        "Reproducibility checks:",
        "  - c1",
        "note",
    ):
        assert fragment in text


def test_to_text_omits_empty_sections() -> None:
    text = Briefing(headline="Head", findings=[], warnings=[], checks=[]).to_text()
    assert text.strip() == "Head"
    assert "Findings:" not in text


# ----------------------------------------------------------------------
# LLM prose and the fallback
# ----------------------------------------------------------------------
def test_llm_prose_degrades_to_empty_when_the_backend_is_absent() -> None:
    """``alphaforge.agents._llm`` is an optional dependency that is not bundled."""
    cp = _copilot(provider="openai")
    assert cp._llm_prose({"model": _result("model", {"a": 1})}) == ""


def test_llm_prose_returns_the_backend_output_when_it_is_there(monkeypatch) -> None:
    fake = types.ModuleType("alphaforge.agents._llm")
    fake.llm_complete = lambda provider, model, prompt: f"{provider}:{len(prompt)}"
    monkeypatch.setitem(sys.modules, "alphaforge.agents._llm", fake)
    cp = _copilot(provider="anthropic", model="claude-x")
    got = cp._llm_prose({"model": _result("model", {"a": 1})})
    assert got.startswith("anthropic:")
    assert int(got.split(":")[1]) > 0


def test_the_structured_prompt_lists_only_successful_tools() -> None:
    results = {
        "model": _result("model", {"rank_ic_mean": 0.02}),
        "backtest": _result("backtest", {"cagr": 0.1}, ok=False),
    }
    prompt = _copilot()._structured_prompt(results)
    assert "[model]" in prompt
    assert "[backtest]" not in prompt


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------
def test_analyze_without_any_state_still_produces_a_briefing() -> None:
    briefing = _copilot().analyze({})
    assert "backtest not run" in briefing.headline
    assert briefing.findings == []
    assert briefing.warnings == []
    assert briefing.llm_note == ""


def test_analyze_does_not_call_the_llm_on_the_deterministic_provider(monkeypatch) -> None:
    called: list = []
    fake = types.ModuleType("alphaforge.agents._llm")
    fake.llm_complete = lambda *a, **k: called.append(a) or "should not happen"
    monkeypatch.setitem(sys.modules, "alphaforge.agents._llm", fake)
    assert _copilot(provider="none").analyze({}).llm_note == ""
    assert called == []


def test_analyze_attaches_the_llm_prose_when_a_provider_is_configured(
    monkeypatch,
) -> None:
    fake = types.ModuleType("alphaforge.agents._llm")
    fake.llm_complete = lambda provider, model, prompt: "prose from the backend"
    monkeypatch.setitem(sys.modules, "alphaforge.agents._llm", fake)
    assert _copilot(provider="openai").analyze({}).llm_note == "prose from the backend"


def test_analyze_keeps_the_deterministic_brief_when_the_llm_fails(monkeypatch) -> None:
    fake = types.ModuleType("alphaforge.agents._llm")

    def boom(*args, **kwargs):
        raise RuntimeError("credential missing")

    fake.llm_complete = boom
    monkeypatch.setitem(sys.modules, "alphaforge.agents._llm", fake)
    briefing = _copilot(provider="openai").analyze({})
    assert briefing.llm_note == ""
    assert briefing.headline
