"""Schema migrations: discovery, idempotency, and drift detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_analytics.warehouse.engine import reset_engines
from eval_analytics.warehouse.migrate import (
    MigrationChecksumError,
    _split_statements,
    applied_migrations,
    current_version,
    discover_migrations,
    migrate,
    rebuild,
)


def test_migrations_are_discovered_in_order():
    migrations = discover_migrations()
    assert migrations, "no migration files found"
    versions = [m.version for m in migrations]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions), "duplicate migration version"


def test_every_migration_file_has_a_valid_name():
    directory = Path(__file__).resolve().parents[1] / "src" / "eval_analytics" / "warehouse" / "migrations"
    for path in directory.glob("V*.sql"):
        discover_migrations(directory)  # raises on a malformed name
        assert path.name[0] == "V"


def test_statement_splitter_drops_comments_and_trailing_semicolons():
    sql = """
    -- a leading comment
    CREATE TABLE a (x INT);
    -- another comment
    CREATE TABLE b (
        y VARCHAR
    );
    """
    statements = _split_statements(sql)
    assert len(statements) == 2
    assert statements[0].startswith("CREATE TABLE a")
    assert not statements[0].endswith(";")
    assert "comment" not in " ".join(statements)


def test_rebuild_applies_every_migration(tmp_path: Path):
    db = tmp_path / "m.duckdb"
    applied = rebuild(db)
    assert applied == [m.name for m in discover_migrations()]
    assert current_version(db) == discover_migrations()[-1].version


def test_migrate_is_idempotent(tmp_path: Path):
    db = tmp_path / "m.duckdb"
    rebuild(db)
    reset_engines()
    assert migrate(db) == [], "second migrate should apply nothing"
    assert current_version(db) == discover_migrations()[-1].version


def test_editing_an_applied_migration_raises(tmp_path: Path):
    """The guard that makes checked-in migrations trustworthy."""
    source = Path(__file__).resolve().parents[1] / "src" / "eval_analytics" / "warehouse" / "migrations"
    staged = tmp_path / "migrations"
    staged.mkdir()
    for path in source.glob("V*.sql"):
        (staged / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    db = tmp_path / "m.duckdb"
    rebuild(db, directory=staged)
    reset_engines()

    target = staged / "V001__star_dimensions.sql"
    target.write_text(target.read_text(encoding="utf-8") + "\n-- tampered\n", encoding="utf-8")

    with pytest.raises(MigrationChecksumError, match="edited after it was applied"):
        migrate(db, directory=staged)


def test_checksums_are_recorded_for_every_migration(tmp_path: Path):
    db = tmp_path / "m.duckdb"
    rebuild(db)
    applied = applied_migrations(db)
    expected = {m.version: m.checksum for m in discover_migrations()}
    assert applied == expected


def test_star_schema_objects_all_exist(db_path: str):
    from eval_analytics.warehouse import query

    objects = set(
        query(
            db_path,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'",
        )["table_name"]
    )
    required = {
        "dim_model", "dim_experiment", "dim_feature", "dim_slice", "dim_date",
        "fact_evaluation_run", "fact_prediction", "fact_serving_request",
        "fact_drift_measurement",
        "ingestion_run", "ingestion_rejection", "mart_freshness",
        "schema_migrations",
        "v_model_quality_headline", "v_drift_pivot", "v_prediction_slice",
    }
    assert required <= objects, f"missing: {sorted(required - objects)}"


def test_fact_tables_have_declared_grain_constraints(db_path: str):
    """Each fact must still carry its natural-key uniqueness constraint.

    Asserted on the constrained *columns* rather than the constraint name:
    DuckDB rewrites ``CONSTRAINT uq_...`` into an engine-generated name, but
    the column list is the grain and that is what must not drift.
    """
    from eval_analytics.warehouse import query

    expected = {
        "fact_evaluation_run": (
            "experiment_id", "model_id", "slice_id", "date_id", "window_id",
        ),
        "fact_prediction": ("evaluation_run_key", "example_id"),
        "fact_serving_request": (
            "model_id", "quantisation", "batch_size", "input_tokens", "request_index",
        ),
        "fact_drift_measurement": ("feature_id", "window_id", "metric_name"),
    }

    rows = query(
        db_path,
        """
        SELECT table_name, constraint_column_names
        FROM duckdb_constraints()
        WHERE constraint_type = 'UNIQUE'
        """,
    )
    found: dict[str, set[tuple[str, ...]]] = {}
    for _, row in rows.iterrows():
        found.setdefault(row["table_name"], set()).add(tuple(row["constraint_column_names"]))

    for table, columns in expected.items():
        assert columns in found.get(table, set()), (
            f"{table} no longer constrains {columns}; found {found.get(table)}"
        )
