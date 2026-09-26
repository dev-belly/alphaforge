"""Build the public synthetic example, figures and reproduction manifest together.

Run from the repository root: python scripts/publish_sample.py
The published sample intentionally bypasses environment overrides and refuses
any provider other than the bundled synthetic sample.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd

from alphaforge.pipeline import ResearchPipeline
from alphaforge.utils.config import Config, load_yaml
from make_assets import render_assets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "sample"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = Config(raw=load_yaml(ROOT / "configs/default.yaml"))
    if cfg.get("data.provider") != "sample":
        raise ValueError("Public examples must use the bundled synthetic sample")
    seed = int(cfg.get("project.seed"))
    OUT.mkdir(parents=True, exist_ok=True)
    state = ResearchPipeline(cfg).run(report_dir=str(OUT))
    bt = state.backtest
    if bt is None or state.report_path is None or state.model_eval is None:
        raise RuntimeError("Required research stages did not complete")
    render_assets(state)
    pd.DataFrame(
        {
            "net_equity": bt.equity,
            "net_return": bt.returns,
            "benchmark_return": bt.benchmark,
            "cost": bt.costs,
        }
    ).to_csv(OUT / "daily_results.csv", index_label="date")
    (OUT / "config.json").write_text(json.dumps(cfg.raw, indent=2) + "\n")
    metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in bt.metrics.items()}
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str, allow_nan=False) + "\n"
    )
    source_paths = sorted(
        [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), ROOT / "configs/default.yaml"]
    )
    artifacts = [
        OUT / name
        for name in ("research_report.html", "daily_results.csv", "config.json", "metrics.json")
    ]
    artifacts += sorted((ROOT / "assets").glob("*.png"))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "sample (synthetic)",
        "seed": seed,
        "command": "python scripts/publish_sample.py",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "artifact_sha256": {str(p.relative_to(ROOT)): digest(p) for p in artifacts},
        "data_dates": [str(state.panel.dates[0].date()), str(state.panel.dates[-1].date())],
        "backtest_dates": [str(bt.returns.index[0].date()), str(bt.returns.index[-1].date())],
        "symbols": len(state.panel.symbols),
        "limitations": [
            "Synthetic data; not market evidence",
            "One seed and configuration",
            "HTML timestamp changes between runs",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = "\n".join(
        f"| {k} | {v:.6g} |" if isinstance(v, (int, float)) else f"| {k} | {v} |"
        for k, v in metrics.items()
    )
    page = f"""# Reproduce the sample

This is a computed run on **synthetic data**, with seed {seed},
{manifest["symbols"]} symbols and data from {manifest["data_dates"][0]} to {manifest["data_dates"][1]}.
The backtest covers {manifest["backtest_dates"][0]} to {manifest["backtest_dates"][1]}.
All README figures and this report come from the same pipeline state.

[Open the full report](sample/research_report.html) · [Run manifest](sample/manifest.json) ·
[Daily results CSV](sample/daily_results.csv) · [Configuration](sample/config.json) · [Metrics JSON](sample/metrics.json)

## Recorded performance

Values below are decimal ratios unless named otherwise; for example, 0.01 CAGR is 1%.
Sharpe, Sortino, Calmar and information ratio are ratios, not percentages.

| Metric | Value |
| --- | ---: |
{rows}

## Run it again

```bash
pip install -e ".[viz]"
python scripts/publish_sample.py
```

This command reads `configs/default.yaml` directly, without environment overrides,
and refuses a non-synthetic provider. Review the manifest's Python and dependency
versions to reproduce this environment. Exact source file hashes accompany the
[source revision](https://github.com/dev-belly/alphaforge/commit/{manifest["source_revision"]}).

Output files go to `docs/sample/`; README figures go to `assets/`.
For normal research runs with other providers or settings, use the [Quickstart](quickstart.md).
Read the [validation limits](validation.md) before interpreting performance.
"""
    (ROOT / "docs/sample-run.md").write_text(page)
    print(json.dumps({"report": str(state.report_path), "metrics": metrics}, default=str))


if __name__ == "__main__":
    main()
