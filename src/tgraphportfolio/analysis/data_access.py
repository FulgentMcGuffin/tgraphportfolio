"""Thin helpers for listing/loading tables from DuckDB or SQLite files."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from tgraphportfolio.backends.duckdb_backend import DuckDBSource
from tgraphportfolio.backends.sqlite_backend import SQLiteSource


def open_backend(db_path: Path | str, *, read_only: bool = True):
    """Open the appropriate backend based on file extension."""
    path = Path(db_path)
    suffix = path.suffix.lower()
    if suffix == ".duckdb":
        return DuckDBSource(path, read_only=read_only)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return SQLiteSource(path, read_only=read_only)
    raise ValueError(
        f"Unsupported database type {suffix!r}. Use .duckdb or .sqlite/.db."
    )


def list_tables(db_path: Path | str) -> list[str]:
    with open_backend(db_path) as db:
        return db.list_tables()


def list_columns(db_path: Path | str, table: str) -> list[str]:
    with open_backend(db_path) as db:
        schema = db.get_schema(table)
        return [col.name for col in schema.columns]


def load_table(
    db_path: Path | str,
    table: str,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load selected columns from ``table`` (all columns if ``columns`` is None)."""
    with open_backend(db_path) as db:
        if columns:
            quoted = ", ".join(f'"{c}"' for c in columns)
            sql = f'SELECT {quoted} FROM "{table}"'
        else:
            sql = f'SELECT * FROM "{table}"'
        return db.run_query(sql)


def distinct_values(
    db_path: Path | str,
    table: str,
    column: str,
    limit: int = 500,
) -> list[str]:
    """Return distinct stringified values for a filter dropdown."""
    with open_backend(db_path) as db:
        sql = (
            f'SELECT DISTINCT "{column}" AS v FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL '
            f'ORDER BY v LIMIT {int(limit)}'
        )
        frame = db.run_query(sql)
        if frame.is_empty():
            return []
        return [str(v) for v in frame.get_column("v").to_list()]


def column_date_bounds(
    db_path: Path | str,
    table: str,
    column: str,
) -> tuple[date | None, date | None]:
    """Return ``(min_date, max_date)`` for a date/datetime column."""
    with open_backend(db_path) as db:
        sql = (
            f'SELECT MIN("{column}") AS lo, MAX("{column}") AS hi '
            f'FROM "{table}"'
        )
        frame = db.run_query(sql)
        if frame.is_empty():
            return None, None
        lo, hi = frame.row(0)
        return _as_date(lo), _as_date(hi)


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
