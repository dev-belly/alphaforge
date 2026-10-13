from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alphaforge.comparison import ranked_weights, validate_coverage, verify_comparison
from alphaforge.data.assurance import audit_local_dataset, write_mirror_assurance


def source_bundle(root):
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    for name in ("constituents", "fundamentals", "industry", "macro"):
        pd.DataFrame({"unused": []}).to_parquet(root / f"{name}.parquet")
    pd.DataFrame({"date": dates, "symbol": ["A", "A"]}).to_parquet(root / "prices.parquet")
    pd.DataFrame({"date": dates, "value": [0.01, -0.02]}).to_parquet(root / "benchmark.parquet")
    raw = root / "source.txt"
    raw.write_text("fixture")
    write_mirror_assurance(root, {"raw": raw}, {"raw": "https://example.test/fixture"}, 90)


def test_incomplete_real_sources_are_exploratory_and_strict_mode_rejects(tmp_path):
    source_bundle(tmp_path)
    result = audit_local_dataset(tmp_path, "2020-01-01", "2020-01-03")
    assert result["status"] == "exploratory"
    assert len(result["limitations"]) == 3
    with pytest.raises(ValueError, match="Complete-source validation failed"):
        audit_local_dataset(tmp_path, "2020-01-01", "2020-01-03", require_complete_sources=True)


def test_changed_source_bytes_and_missing_benchmark_dates_fail(tmp_path):
    source_bundle(tmp_path)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "value": [0.01]}).to_parquet(
        tmp_path / "benchmark.parquet"
    )
    with pytest.raises(ValueError, match="Source identity changed"):
        audit_local_dataset(tmp_path, "2020-01-01", "2020-01-03")
    raw = tmp_path / "source.txt"
    write_mirror_assurance(tmp_path, {"raw": raw}, {"raw": "https://example.test/fixture"}, 90)
    with pytest.raises(ValueError, match="missing 1 price sessions"):
        audit_local_dataset(tmp_path, "2020-01-01", "2020-01-03")


def test_coverage_rejects_a_method_that_omits_a_hard_forecast():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B"],
            "fold": [0, 0],
            "realised": [0.1, -0.1],
            "prediction": [0.2, -0.2],
        }
    )
    validate_coverage(frame, frame.copy())
    with pytest.raises(ValueError, match="same forecast"):
        validate_coverage(frame, frame.iloc[:1])
    with pytest.raises(ValueError, match="overlapping/duplicate"):
        validate_coverage(frame, pd.concat([frame, frame.iloc[:1]]))


def test_top_names_use_only_eligible_scores_and_symbol_ties():
    day = pd.Timestamp("2020-01-02")
    scores = pd.DataFrame([[100.0, 0.5, 0.5, -1.0]], index=[day], columns=["future", "B", "A", "C"])
    panel = SimpleNamespace(
        universe=pd.DataFrame([[False, True, True, True]], index=[day], columns=scores.columns)
    )
    weights = ranked_weights(panel, scores, 1)(day, None)
    assert weights.to_dict() == {"A": 1.0}
    all_weights = ranked_weights(panel, scores, None)(day, None)
    assert set(all_weights.index) == {"A", "B", "C"}
    assert all_weights.sum() == pytest.approx(1.0)


def test_saved_return_evidence_detects_rehashed_financial_headline(tmp_path):
    saved = Path(__file__).parents[2] / "docs/experiments/real-sp500"
    destination = tmp_path / "comparison"
    shutil.copytree(saved, destination)
    assert verify_comparison(destination)["cases"] == 22
    path = destination / "summary.json"
    summary = json.loads(path.read_text())
    summary["cases"][0]["cagr"] += 0.1
    path.write_text(json.dumps(summary))
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["summary.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cagr disagrees"):
        verify_comparison(destination)
