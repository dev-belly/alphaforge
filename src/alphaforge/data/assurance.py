"""Source declarations and byte-level identity for exploratory local market data.

Hashes establish reproducibility, not the truth of a vendor's financial data.
The bundled mirror builder does not provide complete historical membership or
actual filing availability, so its real-market runs remain exploratory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ("prices", "benchmark", "constituents", "fundamentals", "industry", "macro")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_mirror_assurance(
    root: Path, sources: dict[str, Path], urls: dict[str, str], lag_days: int
):
    if type(lag_days) is not int or lag_days < 0:
        raise ValueError("fundamental lag must be a nonnegative integer")
    payload = {
        "schema_version": 1,
        "data_kind": "public_market_mirror",
        "source_declarations": {
            "universe_history": "current members with entry dates; removed/delisted names incomplete",
            "complete_historical_universe": False,
            "fundamental_availability": "fiscal period end plus fixed lag, not actual filing date",
            "actual_filing_availability": False,
            "fundamental_lag_days": lag_days,
            "market_capitalization": "unavailable; panel substitutes a dollar-volume proxy",
            "actual_market_capitalization": False,
        },
        "source_urls": urls,
        "raw_sha256": {name: file_digest(path) for name, path in sources.items()},
        "processed_sha256": {
            f"{name}.parquet": file_digest(root / f"{name}.parquet") for name in TABLES
        },
        "scope": "Identity and declared source properties; not independent economic validation",
    }
    (root / "DATA_ASSURANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def audit_local_dataset(root: str | Path, start: str, end: str, *, require_complete_sources=False):
    root = Path(root)
    path = root / "DATA_ASSURANCE.json"
    if not path.is_file():
        raise ValueError(
            "Missing DATA_ASSURANCE.json; rebuild the mirror or supply a source contract"
        )
    assurance = json.loads(path.read_text(encoding="utf-8"))
    expected = {f"{name}.parquet" for name in TABLES}
    if set(assurance.get("processed_sha256", {})) != expected:
        raise ValueError("Data assurance has an incomplete table inventory")
    for name, digest in assurance["processed_sha256"].items():
        if file_digest(root / name) != digest:
            raise ValueError(f"Source identity changed: {name}; regenerate the assurance record")
    prices = pd.read_parquet(root / "prices.parquet", columns=["date", "symbol"])
    benchmark = pd.read_parquet(root / "benchmark.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(prices["date"])).unique().sort_values()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if len(dates) == 0 or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Requested data window has no price observations")
    benchmark_dates = pd.DatetimeIndex(pd.to_datetime(benchmark["date"]))
    if (
        benchmark_dates.has_duplicates
        or not np.isfinite(benchmark["value"].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Benchmark dates or values are invalid")
    missing = dates.difference(benchmark_dates)
    if len(missing):
        raise ValueError(
            f"Benchmark is missing {len(missing)} price sessions; first {missing[0].date()}. "
            "Choose a shared window; missing benchmark sessions cannot be forward-filled."
        )
    declaration = assurance["source_declarations"]
    flags = (
        "complete_historical_universe",
        "actual_filing_availability",
        "actual_market_capitalization",
    )
    if any(type(declaration.get(key)) is not bool for key in flags):
        raise ValueError("Source properties require explicit boolean declarations")
    limitations = []
    if not declaration["complete_historical_universe"]:
        limitations.append(
            "Historical membership is incomplete; survivorship/selection bias remains."
        )
    if not declaration["actual_filing_availability"]:
        limitations.append(
            "Financial availability is an assumed lag, not actual publication history."
        )
    if not declaration["actual_market_capitalization"]:
        limitations.append("Size uses a dollar-volume proxy, not measured market capitalization.")
    if require_complete_sources and limitations:
        raise ValueError("Complete-source validation failed: " + " ".join(limitations))
    return {
        **assurance,
        "status": "exploratory" if limitations else "source_contract_checked",
        "effective_start": str(dates.min().date()),
        "effective_end": str(dates.max().date()),
        "requested_start": start,
        "requested_end": end,
        "price_sessions": len(dates),
        "limitations": limitations,
    }
