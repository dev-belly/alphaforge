"""Replay the new frozen experiment and reject coherently rehashed false dates."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.comparison import run_comparison, verify_comparison

ROOT = Path(__file__).parents[2]
RELATIVE = Path("docs/experiments/real-sp500-session-v2")


@pytest.fixture
def frozen(tmp_path):
    target = tmp_path / "refit"
    shutil.copytree(ROOT / RELATIVE, target)
    return target


def rewrite(target, name, record):
    (target / name).write_text(json.dumps(record), encoding="utf-8")
    path = target / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files_sha256"][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_refit_retains_identical_raw_sources_and_supervised_dataset():
    old = json.loads((ROOT / "docs/experiments/real-sp500/summary.json").read_text())
    new = json.loads((ROOT / RELATIVE / "summary.json").read_text())
    assert old["source_assurance"]["raw_sha256"] == new["source_assurance"]["raw_sha256"]
    assert old["dataset_sha256"] == new["dataset_sha256"]
    assert new["protocol"]["walk_forward"]["purge_days"] == new["dataset"]["horizon_sessions"] == 21
    verified = verify_comparison(ROOT / RELATIVE)
    assert verified["cases"] == 22
    assert verified["audited_folds"] == 6
    assert verified["return_dates_per_case"] == 1217


def test_rehashed_false_label_endpoint_fails_replay(frozen):
    audit = json.loads((frozen / "split_audit.json").read_text())
    audit["folds"][0]["last_train_label_end"] = audit["folds"][0]["test_start"]
    rewrite(frozen, "split_audit.json", audit)
    with pytest.raises(ValueError, match="differs from reconstructed"):
        verify_comparison(frozen)


def test_rehashed_forecast_denominator_must_match_audited_samples(frozen):
    summary = json.loads((frozen / "summary.json").read_text())
    summary["dataset"]["oos_forecasts"] += 1
    rewrite(frozen, "summary.json", summary)
    with pytest.raises(ValueError, match="oos_forecasts disagrees with split audit"):
        verify_comparison(frozen)


def test_removing_the_audit_from_a_version_two_manifest_is_not_legacy(frozen):
    path = frozen / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files_sha256"]["split_audit.json"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unexpected comparison artifact inventory"):
        verify_comparison(frozen)


def test_rehashed_underpurge_in_both_metadata_records_is_rejected(frozen):
    audit = json.loads((frozen / "split_audit.json").read_text())
    summary = json.loads((frozen / "summary.json").read_text())
    for config in (audit["walk_forward"], summary["protocol"]["walk_forward"]):
        config["purge_days"] = config["embargo_days"] = 0
    rewrite(frozen, "split_audit.json", audit)
    rewrite(frozen, "summary.json", summary)
    with pytest.raises(ValueError, match="Training label overlaps test"):
        verify_comparison(frozen)


def test_invalid_training_boundary_stops_before_comparison_fits_or_writes(tmp_path, monkeypatch):
    def must_not_fit(*_args, **_kwargs):
        raise AssertionError("An unsafe comparison must fail before fitting")

    monkeypatch.setattr("alphaforge.comparison.AlphaModelPipeline.run", must_not_fit)
    calendar = pd.bdate_range("2018-01-01", "2020-03-31")
    dataset = SimpleNamespace(
        dates=pd.Series(np.repeat(calendar[:-21], 2)), metadata={"horizon": 21}
    )
    state = SimpleNamespace(
        dataset=dataset,
        panel=SimpleNamespace(dates=calendar),
        config={
            "data": {"provider": "sample"},
            "model": {"walk_forward": {"train_years": 1, "purge_days": 0, "embargo_days": 0}},
        },
    )
    destination = tmp_path / "unsafe"
    with pytest.raises(ValueError, match="Training label overlaps test"):
        run_comparison(state, destination)
    assert not destination.exists()


@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_new_evidence_survives_native_git_checkout_conversion(tmp_path, autocrlf):
    repo, checkout = tmp_path / "repo", tmp_path / "checkout"
    repo.mkdir()
    shutil.copytree(ROOT / RELATIVE, repo / RELATIVE)
    shutil.copyfile(ROOT / ".gitattributes", repo / ".gitattributes")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            f"core.autocrlf={autocrlf}",
            "-c",
            "core.eol=crlf",
            "checkout-index",
            "--all",
            "--force",
            f"--prefix={checkout.as_posix()}/",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert verify_comparison(checkout / RELATIVE)["audited_folds"] == 6
