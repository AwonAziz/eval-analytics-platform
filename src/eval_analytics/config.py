"""Filesystem layout and tunable ingest settings.

Every path is resolved relative to the repository root so that the CLI, the
API, the dashboard and CI all agree on where the warehouse and the raw
artifacts live without any environment plumbing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# src/eval_analytics/config.py -> src/eval_analytics -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
GENERATED_DIR = RAW_DIR / "generated"

DUCKDB_PATH = Path(
    os.environ.get("EAP_DUCKDB_PATH", WAREHOUSE_DIR / "eval_analytics.duckdb")
)

#: Upstream telemetry DB produced by the llm-drift-monitor project. This is
#: the only source of genuinely measured rows in the warehouse; it is
#: optional, and the pipeline degrades to synthetic-only if it is absent.
#: Set EAP_TELEMETRY_DB_PATH to an empty value to disable it explicitly.
def _resolve_telemetry_db() -> Path | None:
    raw = os.environ.get("EAP_TELEMETRY_DB_PATH")
    if raw is not None and not raw.strip():
        return None
    return Path(raw) if raw else REPO_ROOT.parent / "llm-drift-monitor" / "data" / "telemetry.db"


TELEMETRY_DB_PATH: Path | None = _resolve_telemetry_db()

#: Document-length bucket edges, in tokens. Shared by the ingest layer (which
#: assigns buckets) and the analytics layer (which groups them), so the two
#: can never disagree about where a bucket boundary sits.
LENGTH_BUCKET_EDGES: tuple[tuple[int, int | None], ...] = (
    (1, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, None),
)

LENGTH_BUCKET_SLICE_TYPE = "length_bucket"


@dataclass(frozen=True)
class Settings:
    duckdb_path: Path = DUCKDB_PATH
    raw_dir: Path = RAW_DIR
    generated_dir: Path = GENERATED_DIR
    telemetry_db_path: Path | None = TELEMETRY_DB_PATH

    @property
    def warehouse_dir(self) -> Path:
        return self.duckdb_path.parent

    def ensure_dirs(self) -> None:
        self.warehouse_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()


def length_bucket_id(n_tokens: int | None) -> str | None:
    """Return the dim_slice id for a document-length bucket.

    ``None`` when the token count is unknown, which keeps unlabeled telemetry
    rows out of the length analysis instead of inventing a bucket for them.
    """
    if n_tokens is None or n_tokens <= 0:
        return None
    for low, high in LENGTH_BUCKET_EDGES:
        if n_tokens >= low and (high is None or n_tokens <= high):
            if high is None:
                return f"len_{low}_{low}_plus"
            return f"len_{low}_{high}"
    return None


def length_bucket_label(slice_id: str) -> str:
    """Human-readable label for a length-bucket slice id."""
    if slice_id.startswith("len_"):
        return slice_id.removeprefix("len_").replace("_", "-")
    return slice_id
