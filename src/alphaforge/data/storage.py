"""Persistence layer: Parquet artefacts + a DuckDB query engine.

Design goals
------------
* **Portable** - Parquet is the system of record. Any tool can read it.
* **Incremental** - :meth:`DataStore.upsert_prices` merges new observations into
  the existing table keyed on ``(date, symbol)`` so that daily refreshes do not
  rewrite history.
* **Swappable** - :class:`PostgresAdapter` shows the seam for a real warehouse;
  nothing else in the codebase touches SQL directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("data.store")

DEFAULT_ROOT = Path("data/processed")


@dataclass
class StorageStats:
    rows_written: int = 0
    rows_total: int = 0
    table: str = ""


class DataStore:
    """Parquet-backed store with an optional DuckDB SQL surface."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, use_duckdb: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.use_duckdb = use_duckdb
        self._con: Any = None
        log.debug(f"DataStore root={self.root} duckdb={use_duckdb}")

    # -- connection ------------------------------------------------------
    @property
    def con(self):
        if not self.use_duckdb:
            return None
        if self._con is None:
            import duckdb

            self._con = duckdb.connect(str(self.root / "alphaforge.duckdb"))
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    # -- paths -----------------------------------------------------------
    def path_for(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    def exists(self, table: str) -> bool:
        return self.path_for(table).exists()

    # -- write -----------------------------------------------------------
    def write(
        self, table: str, df: pd.DataFrame, dedupe_keys: list[str] | None = None
    ) -> StorageStats:
        if df is None or df.empty:
            log.warning(f"Refusing to write empty table '{table}'")
            return StorageStats(table=table)
        if dedupe_keys:
            df = df.drop_duplicates(subset=dedupe_keys, keep="last")
        df.to_parquet(self.path_for(table), index=False)
        log.info(f"Wrote {len(df):,} rows -> {self.path_for(table)}")
        return StorageStats(rows_written=len(df), rows_total=len(df), table=table)

    def upsert_prices(self, df: pd.DataFrame) -> StorageStats:
        """Incremental merge on ``(date, symbol)`` - the daily refresh path."""
        table = "prices"
        keys = ["date", "symbol"]
        if df.empty:
            return StorageStats(table=table)
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=keys, keep="last")
        if self.exists(table):
            old = pd.read_parquet(self.path_for(table))
            old["date"] = pd.to_datetime(old["date"])
            merged = pd.concat([old, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=keys, keep="last")
            merged = merged.sort_values(keys).reset_index(drop=True)
            new_rows = len(merged) - len(old)
        else:
            merged = df.sort_values(keys).reset_index(drop=True)
            new_rows = len(merged)
        merged.to_parquet(self.path_for(table), index=False)
        log.info(f"Upsert prices: +{new_rows:,} new rows (total {len(merged):,})")
        return StorageStats(rows_written=new_rows, rows_total=len(merged), table=table)

    # -- read ------------------------------------------------------------
    def read(self, table: str) -> pd.DataFrame:
        path = self.path_for(table)
        if not path.exists():
            raise FileNotFoundError(f"Table '{table}' not found at {path}")
        return pd.read_parquet(path)

    def query(self, sql: str) -> pd.DataFrame:
        """Run SQL over the registered Parquet views (DuckDB backend only)."""
        if self.con is None:
            raise RuntimeError("DuckDB backend disabled")
        return self.con.execute(sql).fetchdf()

    def register_views(self) -> None:
        """Expose every Parquet artefact as a DuckDB view for ad-hoc research SQL."""
        if self.con is None:
            return
        for path in sorted(self.root.glob("*.parquet")):
            name = path.stem
            self.con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')"
            )
        log.debug(f"Registered DuckDB views for {len(list(self.root.glob('*.parquet')))} tables")

    def tables(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet"))


class PostgresAdapter:
    """Skeleton adapter proving the storage layer is not Parquet-locked.

    Requires SQLAlchemy + psycopg (optional dependencies). Credentials are read
    from the environment only - never stored in the repository.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or __import__("os").environ.get("ALPHAFORGE_POSTGRES_DSN")
        if not self.dsn:
            raise RuntimeError("ALPHAFORGE_POSTGRES_DSN is not configured")

    def write(self, table: str, df: pd.DataFrame) -> None:  # pragma: no cover
        from sqlalchemy import create_engine

        engine = create_engine(self.dsn)
        df.to_sql(table, engine, if_exists="append", index=False)

    def read(self, table: str) -> pd.DataFrame:  # pragma: no cover
        from sqlalchemy import create_engine

        engine = create_engine(self.dsn)
        return pd.read_sql_table(table, engine)


__all__ = ["DataStore", "StorageStats", "PostgresAdapter"]
