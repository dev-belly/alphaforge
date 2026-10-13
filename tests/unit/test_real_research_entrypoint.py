"""The real-data command cannot audit one root while loading another."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts/run_real_research.py"


def test_real_research_help_remains_available_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert "--require-complete-sources" in result.stdout


def test_real_research_rejects_a_foreign_working_directory_before_loading_data(tmp_path):
    foreign_data = tmp_path / "data/processed"
    foreign_data.mkdir(parents=True)
    (foreign_data / "prices.parquet").write_bytes(b"a different dataset must never be read")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--no-json"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode != 0
    assert "from the repository root" in result.stderr
    assert "Real dataset not found" not in result.stderr
