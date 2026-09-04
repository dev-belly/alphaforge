"""Fast tests for the CLI wiring.

``cli.py`` is the advertised entry point (``python -m alphaforge.cli``) yet it
was the only module at 0% coverage, because it is normally driven as a
subprocess that pytest cannot trace. These tests cover its *control flow*
offline: argument parsing, symbol normalisation, the API-server branch, and the
graceful "uvicorn not installed" path. The heavy pipeline itself is replaced
with a fake, so the whole file runs in well under a second - the real end-to-end
run stays covered by the slow integration test.
"""

from __future__ import annotations

import sys

import pytest

import alphaforge.pipeline
from alphaforge.cli import _build_parser, main


class _FakeBacktest:
    @staticmethod
    def summary() -> dict[str, float]:
        return {"cagr": 0.0079, "sharpe": 0.13}


class _FakeBriefing:
    @staticmethod
    def to_text() -> str:
        return "FAKE BRIEFING BODY"


class _FakeState:
    backtest: object | None = _FakeBacktest()
    briefing: object | None = _FakeBriefing()
    report_path: str | None = "/tmp/fake_report.html"


class _FakePipeline:
    """Stands in for ``ResearchPipeline`` and records how it was called."""

    calls: list[dict[str, object]] = []
    configs: list[object] = []

    def __init__(self, config: object) -> None:
        _FakePipeline.configs.append(config)

    def run(self, **kwargs: object) -> _FakeState:
        _FakePipeline.calls.append(kwargs)
        return _FakeState()


@pytest.fixture(autouse=True)
def _reset_pipeline_calls() -> None:
    _FakePipeline.calls = []
    _FakePipeline.configs = []


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    # `main()` does `from alphaforge.pipeline import ResearchPipeline` at call
    # time, so patching the module attribute is enough to intercept it.
    monkeypatch.setattr(alphaforge.pipeline, "ResearchPipeline", _FakePipeline)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def test_parser_defaults_match_the_documented_contract() -> None:
    args = _build_parser().parse_args([])
    assert args.seed == 42
    assert args.report_dir == "research/reports"
    assert args.api_port == 8000
    assert args.serve_api is False
    assert args.persist is False
    assert args.print_briefing is False
    assert args.start is None and args.end is None


def test_parser_reads_every_override_flag() -> None:
    args = _build_parser().parse_args(
        [
            "--config",
            "configs/default.yaml",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--model",
            "ridge",
            "--provider",
            "sample",
            "--symbols",
            "AAPL,MSFT",
            "--report-dir",
            "/tmp/out",
            "--seed",
            "7",
            "--persist",
            "--verbose",
            "--print-briefing",
        ]
    )
    assert args.start == "2024-01-01"
    assert args.model == "ridge"
    assert args.provider == "sample"
    assert args.report_dir == "/tmp/out"
    assert args.seed == 7
    assert args.persist and args.verbose and args.print_briefing


# --------------------------------------------------------------------------
# main() - pipeline branch (pipeline replaced with a fake)
# --------------------------------------------------------------------------
def test_main_normalises_symbols_and_forwards_overrides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_pipeline(monkeypatch)
    rc = main(
        [
            "--symbols",
            " aapl , msft ",
            "--model",
            "ridge",
            "--provider",
            "sample",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
        ]
    )
    assert rc == 0
    assert len(_FakePipeline.calls) == 1
    call = _FakePipeline.calls[0]
    # --provider / --model must reach the config, not just the pipeline kwargs.
    config = _FakePipeline.configs[0]
    assert config.section("data")["provider"] == "sample"
    assert config.section("model")["type"] == "ridge"
    # Ticker casing must be normalised so "aapl" and "AAPL" never diverge.
    assert call["symbols"] == ["AAPL", "MSFT"]
    assert call["model_type"] == "ridge"
    assert call["start"] == "2024-01-01"
    assert call["report_dir"] == "research/reports"
    assert call["persist"] is False

    # The report path is echoed for the human running the command.
    assert "/tmp/fake_report.html" in capsys.readouterr().out


def test_main_leaves_symbols_none_when_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    """No --symbols means "let the provider decide", not an empty universe."""
    _patch_pipeline(monkeypatch)
    assert main([]) == 0
    assert _FakePipeline.calls[0]["symbols"] is None


def test_main_prints_briefing_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_pipeline(monkeypatch)
    main([])
    assert "FAKE BRIEFING BODY" not in capsys.readouterr().out

    main(["--print-briefing"])
    assert "FAKE BRIEFING BODY" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main() - API server branch
# --------------------------------------------------------------------------
def test_serve_api_launches_uvicorn_without_running_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: dict[str, object] = {}

    def _fake_run(target: str, **kwargs: object) -> None:
        started["target"] = target
        started.update(kwargs)

    # uvicorn is an optional (api-extra) dependency: skip rather than error.
    uvicorn = pytest.importorskip("uvicorn")
    monkeypatch.setattr(uvicorn, "run", _fake_run)
    rc = main(["--serve-api", "--api-port", "8123"])

    assert rc == 0
    assert started["target"] == "alphaforge_api.main:app"
    assert started["port"] == 8123
    assert started["host"] == "127.0.0.1"
    # Serving must not kick off a research run.
    assert _FakePipeline.calls == []


def test_serve_api_degrades_when_uvicorn_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional dependency returns an error code, not a traceback."""
    # `sys.modules[name] = None` makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert main(["--serve-api"]) == 1
