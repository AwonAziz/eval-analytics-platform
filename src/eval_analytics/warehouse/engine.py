"""Warehouse access: DuckDB engine plus SQLAlchemy/Core query helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_ENGINE_CACHE: dict[str, Engine] = {}


def get_engine(db_path: Path | str) -> Engine:
    """Return a cached SQLAlchemy engine for a DuckDB file.

    DuckDB allows only one handle per database file per process, and refuses
    to open a second one with a different configuration -- so every caller
    shares a single read-write engine per path rather than opening a
    read-only one for queries. DuckDB happily serves reads on a read-write
    handle.
    """
    path = Path(db_path)
    key = path.as_posix()
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = create_engine(
            f"duckdb:///{path.as_posix()}",
            future=True,
        )
    return _ENGINE_CACHE[key]


def require_warehouse(db_path: Path | str) -> Path:
    """Assert the warehouse has been built, and return its path."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"warehouse not found at {path}; run `eap build` to create it"
        )
    return path


def reset_engines() -> None:
    """Dispose every cached engine. Used by tests that rebuild the warehouse."""
    for engine in _ENGINE_CACHE.values():
        engine.dispose()
    _ENGINE_CACHE.clear()


def query(db_path: Path | str, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a read query and return a DataFrame."""
    engine = get_engine(require_warehouse(db_path))
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params or {})


def execute(db_path: Path | str, sql: str, params: dict[str, Any] | None = None) -> None:
    """Run a write statement."""
    engine = get_engine(db_path)
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def execute_many(db_path: Path | str, sql: str, rows: list[dict[str, Any]]) -> None:
    """Run a write statement once per row, inside one transaction."""
    if not rows:
        return
    engine = get_engine(db_path)
    with engine.begin() as conn:
        conn.execute(text(sql), rows)
