"""Verify that stored model fold scores appear in the browser snapshot."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rank_ic_joins_each_model_and_seed_to_its_saved_windows():
    directory = ROOT / "docs"
    with (directory / "experiments/real-sp500-session-v2/folds.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        expected = {
            (row["method"], int(row["seed"]), int(row["fold"]) + 1): float(row["rank_ic_mean"])
            for row in csv.DictReader(stream)
        }
    script = (directory / "workbench/data.js").read_text(encoding="utf-8")
    data = json.loads(script.split("window.workbenchData = ", 1)[1].removesuffix(";\n"))
    count = 0
    for entry in data["records"]:
        method = entry["values"]["method"]
        seed = entry["values"]["seed"]
        rows = entry["charts"]["folds"]
        if seed == "基准":
            assert rows == []
            continue
        assert [row["value"] for row in rows] == [
            expected[method, int(seed), fold] for fold in range(1, 7)
        ]
        count += 1
    assert count == 18
