"""Deterministic surrogate keys for fact tables.

DuckDB's built-in ``hash()`` is only guaranteed stable within a release
series, which would silently change every fact primary key on a dependency
bump. These helpers use BLAKE2b over the natural grain instead, so a rebuild
from identical inputs produces byte-identical keys and a diff of the
warehouse is a meaningful review artifact.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

_MASK64 = (1 << 63) - 1


def stable_key(*parts: object) -> int:
    """Hash a natural key tuple into a non-negative signed 64-bit integer."""
    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK64


def stable_keys(rows: Iterable[Iterable[object]]) -> list[int]:
    """Vectorised :func:`stable_key` over a sequence of natural keys."""
    return [stable_key(*row) for row in rows]
