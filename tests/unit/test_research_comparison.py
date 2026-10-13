from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alphaforge.comparison import ranked_weights, validate_coverage, verify_comparison
from alphaforge.data.assurance import audit_local_dataset, write_mirror_assurance

HEADLINES = (
    "cagr",
    "sharpe",
    "max_drawdown",
    "ann_vol",
    "total_return",
    "gross_cagr",
    "gross_sharpe",
    "gross_max_drawdown",
    "gross_ann_vol",
    "gross_total_return",
)


@pytest.fixture
def saved_comparison(tmp_path):
    saved = Path(__file__).parents[2] / "docs/experiments/real-sp500"
    destination = tmp_path / "comparison"
    shutil.copytree(saved, destination)
    return destination


def rehash_artifact(directory, name):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files_sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def write_summary(directory, summary):
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rehash_artifact(directory, "summary.json")


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


@pytest.mark.parametrize("autocrlf", ["false", "true"])
@pytest.mark.parametrize("eol", ["lf", "crlf"])
def test_frozen_evidence_survives_git_checkout_line_ending_settings(tmp_path, autocrlf, eol):
    root = Path(__file__).parents[2]
    relative = Path("docs/experiments/real-sp500")
    repository = tmp_path / "repository"
    checkout = tmp_path / "checkout"
    repository.mkdir()
    shutil.copytree(root / relative, repository / relative)
    shutil.copyfile(root / ".gitattributes", repository / ".gitattributes")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "add", "."],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            f"core.autocrlf={autocrlf}",
            "-c",
            f"core.eol={eol}",
            "checkout-index",
            "--all",
            "--force",
            f"--prefix={checkout.as_posix()}/",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    manifest = json.loads((root / relative / "manifest.json").read_text())
    for name in ("manifest.json", *manifest["files_sha256"]):
        assert (checkout / relative / name).read_bytes() == (root / relative / name).read_bytes()
    assert verify_comparison(checkout / relative)["cases"] == 22


def test_saved_evidence_rejects_line_ending_changes_without_rehashing(tmp_path):
    saved = Path(__file__).parents[2] / "docs/experiments/real-sp500"
    destination = tmp_path / "comparison"
    shutil.copytree(saved, destination)
    path = destination / "folds.csv"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="artifact digest mismatch: folds.csv"):
        verify_comparison(destination)


@pytest.mark.parametrize("title", HEADLINES)
def test_rehashed_nan_metrics_csv_cannot_bypass_headline_comparison(saved_comparison, title):
    path = saved_comparison / "metrics.csv"
    metrics = pd.read_csv(path)
    metrics.loc[0, title] = float("nan")
    metrics.to_csv(path, index=False, lineterminator="\n")
    rehash_artifact(saved_comparison, "metrics.csv")
    with pytest.raises(ValueError, match=f"{title} must be finite"):
        verify_comparison(saved_comparison)


@pytest.mark.parametrize("title", HEADLINES)
def test_rehashed_missing_case_headline_cannot_skip_verification(saved_comparison, title):
    summary = json.loads((saved_comparison / "summary.json").read_text(encoding="utf-8"))
    del summary["cases"][0][title]
    write_summary(saved_comparison, summary)
    with pytest.raises(ValueError, match=f"Missing required comparison headline: {title}"):
        verify_comparison(saved_comparison)


@pytest.mark.parametrize("metric", ["cagr", "sharpe", "max_drawdown"])
@pytest.mark.parametrize("stat", ["mean", "std"])
def test_rehashed_nan_aggregates_cannot_bypass_comparison(saved_comparison, metric, stat):
    summary = json.loads((saved_comparison / "summary.json").read_text(encoding="utf-8"))
    summary["aggregates"][0][f"{metric}_{stat}"] = float("nan")
    write_summary(saved_comparison, summary)
    with pytest.raises(ValueError, match="aggregates must be finite"):
        verify_comparison(saved_comparison)


def test_rehashed_summary_must_match_manifest_dataset_identity(saved_comparison):
    summary = json.loads((saved_comparison / "summary.json").read_text(encoding="utf-8"))
    summary["dataset_sha256"] = "0" * 64
    write_summary(saved_comparison, summary)
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        verify_comparison(saved_comparison)


def test_rehashed_duplicate_deterministic_seed_cannot_replace_a_case(saved_comparison):
    summary = json.loads((saved_comparison / "summary.json").read_text(encoding="utf-8"))
    indices = [
        index
        for index, case in enumerate(summary["cases"])
        if case["method"] == "ridge" and case["cost_multiplier"] == 1.0
    ]
    summary["cases"][indices[1]] = copy.deepcopy(summary["cases"][indices[0]])
    write_summary(saved_comparison, summary)
    with pytest.raises(ValueError, match="case coverage mismatch"):
        verify_comparison(saved_comparison)


def test_rehashed_unused_metrics_row_is_rejected(saved_comparison):
    path = saved_comparison / "metrics.csv"
    metrics = pd.read_csv(path)
    extra = metrics.iloc[[0]].copy()
    extra["seed"] = 777
    pd.concat([metrics, extra], ignore_index=True).to_csv(path, index=False, lineterminator="\n")
    rehash_artifact(saved_comparison, "metrics.csv")
    with pytest.raises(ValueError, match="metrics CSV coverage mismatch"):
        verify_comparison(saved_comparison)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_rehashed_aggregate_inventory_must_match_all_cases(saved_comparison, change):
    summary = json.loads((saved_comparison / "summary.json").read_text(encoding="utf-8"))
    if change == "missing":
        summary["aggregates"].pop()
    else:
        summary["aggregates"].append(copy.deepcopy(summary["aggregates"][0]))
    write_summary(saved_comparison, summary)
    with pytest.raises(ValueError, match="aggregate coverage mismatch"):
        verify_comparison(saved_comparison)
