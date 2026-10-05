"""Shared fixtures.

The warehouse build is slow (it lands and validates ~25k rows), so it happens
once per session and every test reads the same file.

Fixtures are hermetic by default: the telemetry database is not consulted, so
the suite passes on a machine that has never seen the upstream project. One
test opts back in and skips when the database is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_analytics.config import Settings
from eval_analytics.ingest.pipeline import run_pipeline
from eval_analytics.warehouse import reset_engines


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def warehouse_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("warehouse")


@pytest.fixture(scope="session")
def settings(warehouse_dir: Path) -> Settings:
    """Hermetic settings: generated artifacts only, no telemetry database."""
    return Settings(
        duckdb_path=warehouse_dir / "test.duckdb",
        raw_dir=warehouse_dir / "raw",
        generated_dir=warehouse_dir / "raw" / "generated",
        telemetry_db_path=None,
    )


@pytest.fixture(scope="session")
def built(settings: Settings):
    """Build the warehouse once and return the build report."""
    report = run_pipeline(
        db_path=settings.duckdb_path, settings=settings, verbose=False
    )
    assert report.status == "success", report.summary()
    return report


@pytest.fixture(scope="session")
def db_path(built, settings: Settings) -> str:
    return str(settings.duckdb_path)


@pytest.fixture(scope="session")
def telemetry_settings(warehouse_dir: Path) -> Settings | None:
    """Settings that include the real telemetry database, if it exists."""
    from eval_analytics.config import TELEMETRY_DB_PATH

    if not TELEMETRY_DB_PATH.exists():
        return None
    return Settings(
        duckdb_path=warehouse_dir / "telemetry.duckdb",
        raw_dir=warehouse_dir / "raw_telemetry",
        generated_dir=warehouse_dir / "raw_telemetry" / "generated",
        telemetry_db_path=TELEMETRY_DB_PATH,
    )


@pytest.fixture(autouse=True)
def _clean_engines():
    """Engines cache a DuckDB handle per file; close them between tests."""
    yield
    reset_engines()
