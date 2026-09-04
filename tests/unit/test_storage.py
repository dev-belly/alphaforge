"""Offline tests for the Parquet/DuckDB persistence layer (``data/storage.py``).

Parquet is the system of record, so a bug here silently corrupts every downstream
artefact. These tests lock the contracts that matter, with **no network and no
shared state** (every store lives in a ``tmp_path``):

  * :meth:`DataStore.write` refuses empty frames instead of truncating a table.
  * :meth:`DataStore.upsert_prices` merges keyed on ``(date, symbol)`` — the daily
    refresh path. New observations append; a *correction* to an existing key
    replaces the old value **without duplicating the row**. Getting this wrong
    either rewrites history or double-counts a bar, which is exactly the class of
    defect this suite exists to catch.
  * The DuckDB surface is optional: with it disabled the SQL entry points must
    fail loudly rather than return nothing.

DuckDB-specific behaviour is guarded with ``importorskip`` so the suite still runs
where that optional dependency is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data.storage import DEFAULT_ROOT, DataStore, PostgresAdapter, StorageStats


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    """A store with the DuckDB surface disabled (pure Parquet, always available)."""
    return DataStore(root=tmp_path / "store", use_duckdb=False)


def _prices(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "symbol", "close"])


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------


def test_default_root_is_a_path() -> None:
    assert isinstance(DEFAULT_ROOT, Path)


def test_storage_stats_defaults_to_zero_rows() -> None:
    stats = StorageStats()
    assert (stats.rows_written, stats.rows_total, stats.table) == (0, 0, "")


def test_store_creates_its_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "processed"
    DataStore(root=root, use_duckdb=False)
    assert root.is_dir()


def test_path_for_and_exists_track_the_same_file(store: DataStore) -> None:
    assert store.path_for("prices") == store.root / "prices.parquet"
    assert not store.exists("prices")
    store.write("prices", _prices([("2024-01-02", "AAPL", 100.0)]))
    assert store.exists("prices")


def test_tables_lists_written_artifacts_sorted(store: DataStore) -> None:
    store.write("prices", _prices([("2024-01-02", "AAPL", 100.0)]))
    store.write("fundamentals", _prices([("2024-01-02", "AAPL", 1.0)]))
    assert store.tables() == ["fundamentals", "prices"]


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------


def test_write_round_trips_through_read(store: DataStore) -> None:
    df = _prices([("2024-01-02", "AAPL", 100.0), ("2024-01-03", "AAPL", 101.0)])
    stats = store.write("prices", df)

    assert stats.rows_written == 2
    assert stats.rows_total == 2
    assert stats.table == "prices"
    pd.testing.assert_frame_equal(store.read("prices"), df)


@pytest.mark.parametrize("empty", [pd.DataFrame(), pd.DataFrame(columns=["date", "symbol"])])
def test_write_refuses_empty_frame(store: DataStore, empty: pd.DataFrame) -> None:
    """Writing an empty frame must not truncate an existing table."""
    stats = store.write("prices", empty)
    assert stats.rows_written == 0
    assert stats.table == "prices"
    assert not store.exists("prices")


def test_write_refuses_none(store: DataStore) -> None:
    stats = store.write("prices", None)  # type: ignore[arg-type]
    assert stats.rows_written == 0
    assert not store.exists("prices")


def test_write_dedupes_on_keys_keeping_last(store: DataStore) -> None:
    df = _prices(
        [
            ("2024-01-02", "AAPL", 100.0),
            ("2024-01-02", "AAPL", 111.0),  # corrected bar, later in the frame
            ("2024-01-03", "AAPL", 102.0),
        ]
    )
    stats = store.write("prices", df, dedupe_keys=["date", "symbol"])

    out = store.read("prices")
    assert stats.rows_written == 2
    assert len(out) == 2
    # ``write`` persists exactly what it is given (only ``upsert_prices``
    # normalises dates), so the raw string key is what comes back here.
    assert out.loc[out["date"] == "2024-01-02", "close"].iloc[0] == 111.0


def test_write_without_keys_keeps_duplicates(store: DataStore) -> None:
    df = _prices([("2024-01-02", "AAPL", 100.0), ("2024-01-02", "AAPL", 111.0)])
    store.write("prices", df)
    assert len(store.read("prices")) == 2  # no dedupe requested -> preserved


def test_read_missing_table_raises(store: DataStore) -> None:
    with pytest.raises(FileNotFoundError, match="Table 'prices' not found"):
        store.read("prices")


# --------------------------------------------------------------------------
# upsert_prices - the incremental refresh path
# --------------------------------------------------------------------------


def test_upsert_creates_table_when_absent(store: DataStore) -> None:
    df = _prices([("2024-01-03", "AAPL", 3.0), ("2024-01-02", "AAPL", 2.0)])
    stats = store.upsert_prices(df)

    out = store.read("prices")
    assert stats.rows_written == 2
    assert stats.rows_total == 2
    # Rows are persisted in (date, symbol) order regardless of input order.
    assert out["close"].tolist() == [2.0, 3.0]


def test_upsert_appends_new_observations_only(store: DataStore) -> None:
    store.upsert_prices(_prices([("2024-01-02", "AAPL", 1.0), ("2024-01-03", "AAPL", 2.0)]))
    stats = store.upsert_prices(_prices([("2024-01-04", "AAPL", 3.0), ("2024-01-02", "MSFT", 9.0)]))

    out = store.read("prices")
    assert stats.rows_written == 2  # only the genuinely new rows
    assert stats.rows_total == 4
    assert len(out) == 4


def test_upsert_correction_replaces_value_without_duplicating_row(store: DataStore) -> None:
    """The core contract: re-ingesting a known key corrects it, never duplicates it.

    A vendor restatement must update the bar in place. If this regressed, the
    store would accumulate duplicate (date, symbol) rows and every downstream
    return series would double-count that day.
    """
    store.upsert_prices(_prices([("2024-01-02", "AAPL", 100.0), ("2024-01-03", "AAPL", 101.0)]))
    stats = store.upsert_prices(_prices([("2024-01-02", "AAPL", 100.5)]))

    out = store.read("prices")
    assert stats.rows_written == 0  # a correction adds no rows
    assert stats.rows_total == 2
    assert len(out) == 2
    assert out.loc[out["date"] == pd.Timestamp("2024-01-02"), "close"].iloc[0] == 100.5
    # The untouched bar is preserved - history is not rewritten.
    assert out.loc[out["date"] == pd.Timestamp("2024-01-03"), "close"].iloc[0] == 101.0


def test_upsert_is_idempotent(store: DataStore) -> None:
    """Re-running the same daily refresh must not change the table."""
    df = _prices([("2024-01-02", "AAPL", 1.0), ("2024-01-03", "AAPL", 2.0)])
    store.upsert_prices(df)
    first = store.read("prices")

    store.upsert_prices(df)
    pd.testing.assert_frame_equal(store.read("prices"), first)


def test_upsert_empty_frame_is_a_noop(store: DataStore) -> None:
    stats = store.upsert_prices(_prices([]))
    assert stats.rows_written == 0
    assert not store.exists("prices")


def test_upsert_coerces_string_dates_to_datetime(store: DataStore) -> None:
    store.upsert_prices(_prices([("2024-01-02", "AAPL", 1.0)]))
    out = store.read("prices")
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02")


# --------------------------------------------------------------------------
# Optional DuckDB surface: disabled -> fail loudly; enabled -> works
# --------------------------------------------------------------------------


def test_query_raises_when_duckdb_disabled(store: DataStore) -> None:
    assert store.con is None
    with pytest.raises(RuntimeError, match="DuckDB backend disabled"):
        store.query("SELECT 1")


def test_register_views_and_close_are_safe_without_duckdb(store: DataStore) -> None:
    store.write("prices", _prices([("2024-01-02", "AAPL", 1.0)]))
    store.register_views()  # no-op, must not raise
    store.close()
    assert store.con is None


def test_duckdb_connection_is_lazy_and_cached(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    duck_store = DataStore(root=tmp_path / "duck", use_duckdb=True)

    con = duck_store.con
    assert con is not None
    assert duck_store.con is con  # created once, then reused
    duck_store.close()
    assert duck_store._con is None


def test_duckdb_query_reads_registered_parquet_views(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    duck_store = DataStore(root=tmp_path / "duck", use_duckdb=True)
    duck_store.upsert_prices(_prices([("2024-01-02", "AAPL", 1.0), ("2024-01-03", "AAPL", 2.0)]))

    duck_store.register_views()
    out = duck_store.query("SELECT COUNT(*) AS n FROM prices")

    assert int(out["n"].iloc[0]) == 2
    duck_store.close()


# --------------------------------------------------------------------------
# PostgresAdapter: the seam proving storage is not Parquet-locked
# --------------------------------------------------------------------------


def test_postgres_adapter_requires_a_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHAFORGE_POSTGRES_DSN", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAFORGE_POSTGRES_DSN is not configured"):
        PostgresAdapter()


def test_postgres_adapter_reads_dsn_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAFORGE_POSTGRES_DSN", "postgresql://user@host/db")
    assert PostgresAdapter().dsn == "postgresql://user@host/db"


def test_postgres_adapter_accepts_explicit_dsn() -> None:
    assert PostgresAdapter(dsn="postgresql://explicit/db").dsn == "postgresql://explicit/db"
