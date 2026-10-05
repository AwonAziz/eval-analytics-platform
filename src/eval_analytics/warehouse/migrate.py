"""Schema migration runner.

Migrations are plain SQL files named ``V###__description.sql`` in
``migrations/``. They are applied in version order inside a single
transaction each, and the applied checksum is recorded.

Two deliberate properties:

* **Idempotent** — every statement is written to be safely re-runnable, and
  an already-applied migration is skipped.
* **Drift-detecting** — if a migration file's contents change after it was
  applied, the runner raises instead of silently diverging from what CI
  built. This is the check that makes "real schema migrations checked into
  the repo" enforceable rather than aspirational.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from .engine import get_engine, reset_engines

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_FILENAME_RE = re.compile(r"^V(?P<version>\d+)__(?P<slug>.+)\.sql$")

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR PRIMARY KEY,
    slug        VARCHAR NOT NULL,
    checksum    VARCHAR NOT NULL,
    applied_at  TIMESTAMP NOT NULL,
    duration_ms DOUBLE
)
"""


@dataclass(frozen=True)
class Migration:
    version: str
    slug: str
    sql: str
    checksum: str

    @property
    def name(self) -> str:
        return f"V{self.version}__{self.slug}"


class MigrationChecksumError(RuntimeError):
    """Raised when an applied migration file has been edited in place."""


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Load and order every migration file."""
    if not directory.exists():
        return []
    found: list[Migration] = []
    for path in sorted(directory.glob("V*.sql")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            raise ValueError(f"malformed migration filename: {path.name}")
        sql = path.read_text(encoding="utf-8")
        found.append(
            Migration(
                version=match.group("version"),
                slug=match.group("slug"),
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    return sorted(found, key=lambda m: m.version)


def applied_migrations(db_path: Path | str) -> dict[str, str]:
    """Map of version -> checksum for migrations already applied."""
    engine = get_engine(db_path)
    with engine.begin() as conn:
        conn.execute(text(_BOOTSTRAP))
        rows = conn.execute(text("SELECT version, checksum FROM schema_migrations")).fetchall()
    return {row[0]: row[1] for row in rows}


def migrate(db_path: Path | str, directory: Path = MIGRATIONS_DIR, verbose: bool = False) -> list[str]:
    """Apply all pending migrations. Returns the names of those applied."""
    migrations = discover_migrations(directory)
    already = applied_migrations(db_path)

    for migration in migrations:
        recorded = already.get(migration.version)
        if recorded is not None:
            if recorded != migration.checksum:
                raise MigrationChecksumError(
                    f"migration V{migration.version}__{migration.slug} was edited after it was "
                    f"applied (recorded {recorded[:12]}, on disk {migration.checksum[:12]}). "
                    "Add a new migration instead of rewriting an applied one."
                )
            continue

        engine = get_engine(db_path)
        started = time.perf_counter()
        with engine.begin() as conn:
            for statement in _split_statements(migration.sql):
                conn.execute(text(statement))
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, slug, checksum, applied_at, duration_ms) "
                    "VALUES (:v, :s, :c, CAST(now() AS TIMESTAMP), :d)"
                ),
                {
                    "v": migration.version,
                    "s": migration.slug,
                    "c": migration.checksum,
                    "d": round((time.perf_counter() - started) * 1000.0, 3),
                },
            )
        if verbose:
            print(f"  applied {migration.name}")
    return [m.name for m in migrations if m.version not in already]


def current_version(db_path: Path | str) -> str | None:
    applied = applied_migrations(db_path)
    return max(applied) if applied else None


def rebuild(db_path: Path | str, directory: Path = MIGRATIONS_DIR, verbose: bool = False) -> list[str]:
    """Drop the warehouse file and re-apply every migration from scratch.

    Returns the list of migrations that were applied.
    """
    db_path = Path(db_path)
    reset_engines()
    if db_path.exists():
        db_path.unlink()
    wal = db_path.with_suffix(db_path.suffix + ".wal")
    if wal.exists():
        wal.unlink()
    return migrate(db_path, directory, verbose=verbose)


def _split_statements(sql: str) -> list[str]:
    """Split a migration file into executable statements.

    Handles the ``;``-terminated statement style used throughout, including
    the trailing comment block, without pulling in a full SQL parser.
    """
    statements: list[str] = []
    buffer: list[str] = []
    for raw_line in sql.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--"):
            continue
        buffer.append(raw_line)
        if line.endswith(";"):
            statements.append("\n".join(buffer).rstrip().rstrip(";").strip())
            buffer = []
    tail = "\n".join(buffer).strip()
    if tail:
        statements.append(tail)
    return [s for s in statements if s.strip()]
