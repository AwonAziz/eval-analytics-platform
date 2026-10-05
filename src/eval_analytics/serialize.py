"""JSON-safe normalisation for query results.

Query results come back as pandas / numpy types, and ``json.dumps`` refuses
them. The default escape hatch is ``default=str``, which silently turns
integers into strings -- a consumer doing arithmetic on ``n_paired`` then gets
a ``TypeError`` far from the cause.

Both the CLI and the API normalise through :func:`jsonable` so a number is a
number on either surface.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any


def jsonable(value: Any) -> Any:
    """Recursively convert a result object into JSON-serialisable types."""
    if isinstance(value, float):
        # NaN and Infinity are not valid JSON; null is the honest encoding.
        return None if math.isnan(value) or math.isinf(value) else value
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # numpy / pandas scalars expose .item(); pandas NaT and NA need care.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except (TypeError, ValueError):
            pass

    if hasattr(value, "isoformat"):
        return value.isoformat()

    try:
        import pandas as pd

        if value is pd.NaT:
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError, ImportError):
        pass

    return str(value)
