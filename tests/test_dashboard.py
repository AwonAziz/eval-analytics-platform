"""Every dashboard page must render without raising.

Streamlit swallows exceptions into the page, so a broken chart is invisible
until someone opens the app. Rendering each view through ``AppTest`` turns
that into a test failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def app_test(db_path: str):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    os.environ["EAP_DUCKDB_PATH"] = db_path
    return streamlit_testing.AppTest.from_file(
        str(DASHBOARD), default_timeout=600
    )


def _labels(app_test) -> list[str]:
    return app_test.sidebar.radio[0].options


def test_dashboard_renders_every_view(app_test):
    app_test.run()
    assert not app_test.exception, [str(e.value) for e in app_test.exception]

    views = _labels(app_test)
    assert views == [
        "Overview",
        "Arm comparison",
        "Calibration",
        "Quantisation",
        "Drift",
        "Length buckets",
        "Slices",
    ]

    for view in views:
        app_test.sidebar.radio[0].set_value(view).run()
        assert not app_test.exception, (
            f"{view} raised: {[str(e.value) for e in app_test.exception]}"
        )


def test_every_view_produces_at_least_one_chart(app_test):
    app_test.run()
    for view in _labels(app_test):
        app_test.sidebar.radio[0].set_value(view).run()
        if view == "Overview":
            # Overview is a data table, not a chart page.
            continue
        assert len(app_test.get("plotly_chart")) > 0, f"{view} rendered no charts"


def test_analytical_views_show_the_headline_numbers(app_test):
    app_test.run()
    app_test.sidebar.radio[0].set_value("Arm comparison").run()
    values = {m.label: m.value for m in app_test.metric}
    assert "Paired examples" in values
    assert values["Paired examples"] == "2,400"
    assert "McNemar p" in values


def test_length_view_identifies_the_worst_bucket(app_test):
    app_test.run()
    app_test.sidebar.radio[0].set_value("Length buckets").run()
    values = {m.label: m.value for m in app_test.metric}
    assert any("worst bucket" in label for label in values)
    assert any(str(v) == "17-32" for v in values.values())
