"""FastAPI surface and the read-only SQL guard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eval_analytics.api.app import create_app


@pytest.fixture(scope="module")
def client(db_path: str):
    return TestClient(create_app(db_path))


class TestHealth:
    def test_health_reports_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_unknown_warehouse_is_unavailable_not_a_crash(self, tmp_path):
        response = TestClient(create_app(tmp_path / "nope.duckdb")).get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "unavailable"


class TestMetadata:
    @pytest.mark.parametrize(
        "url",
        ["/meta/dimensions", "/meta/arms", "/meta/summary", "/query/runs"],
    )
    def test_metadata_endpoints_return_json(self, client: TestClient, url: str):
        response = client.get(url)
        assert response.status_code == 200
        assert isinstance(response.json(), dict)

    def test_dimensions_include_every_dim_table(self, client: TestClient):
        payload = client.get("/meta/dimensions").json()
        assert set(payload) == {"models", "experiments", "features", "slices"}
        assert payload["models"]

    def test_quality_gate_endpoint_is_200_when_clean(self, client: TestClient):
        response = client.get("/meta/quality-gate")
        assert response.status_code == 200
        assert response.json()["passed"] is True


class TestAnalyticalEndpoints:
    @pytest.mark.parametrize(
        "url",
        [
            "/analysis/lora-vs-full?n_boot=200",
            "/analysis/calibration",
            "/analysis/quantisation",
            "/analysis/drift",
            "/analysis/length-buckets",
            "/analysis/slice-losses?n_boot=200",
        ],
    )
    def test_every_analysis_responds(self, client: TestClient, url: str):
        response = client.get(url)
        assert response.status_code == 200, response.text
        assert response.json()

    def test_lora_vs_full_shape(self, client: TestClient):
        payload = client.get("/analysis/lora-vs-full?n_boot=200").json()
        assert payload["n_examples"] > 0
        assert "overall" in payload and "per_class" in payload
        assert payload["overall"]["ci_low"] <= payload["overall"]["delta"]

    def test_calibration_exposes_bucket_contributions(self, client: TestClient):
        payload = client.get("/analysis/calibration").json()
        assert payload["arms"]
        for entry in payload["arms"]:
            assert entry["buckets"]
            for bucket in entry["buckets"]:
                assert bucket["ece_contribution"] >= 0
                assert 0.0 <= bucket["share_of_ece"] <= 1.0

    def test_drift_can_be_filtered(self, client: TestClient):
        payload = client.get("/analysis/drift?feature=upper_ratio").json()
        assert len(payload["features"]) == 1
        assert payload["features"][0]["feature_id"] == "upper_ratio"

    def test_invalid_arm_pair_is_a_400_not_a_500(self, client: TestClient):
        response = client.get("/analysis/lora-vs-full?arm_b=nope")
        assert response.status_code == 400


class TestSqlEndpoint:
    def test_select_is_allowed(self, client: TestClient):
        response = client.post(
            "/sql",
            json={"sql": "SELECT slice_type, COUNT(*) AS n FROM dim_slice GROUP BY 1"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["row_count"] > 0
        assert "slice_type" in payload["columns"]

    def test_with_cte_is_allowed(self, client: TestClient):
        response = client.post(
            "/sql",
            json={"sql": "WITH t AS (SELECT 1 AS x) SELECT * FROM t"},
        )
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM fact_prediction",
            "INSERT INTO dim_model VALUES ('a','b','none','m','fp32',NULL,NULL,NULL,NULL,NULL,NULL,FALSE,'real')",
            "DROP TABLE fact_prediction",
            "UPDATE dim_model SET arm = 'lora'",
            "CREATE TABLE evil (x INT)",
            "ALTER TABLE dim_model ADD COLUMN x INT",
            "COPY fact_prediction TO 'out.csv'",
            "ATTACH 'other.duckdb'",
            "PRAGMA database_list",
            "SET threads = 1",
            "INSTALL httpfs",
            "SHOW TABLES",
            "EXPLAIN SELECT 1",
            "VACUUM",
            "SELECT 1; DROP TABLE dim_slice",
            "SELECT 1; SELECT 2",
        ],
    )
    def test_write_and_ddl_are_blocked(self, client: TestClient, sql: str):
        response = client.post("/sql", json={"sql": sql})
        assert response.status_code == 400, f"{sql!r} was not blocked"
        assert "error" in response.json() or "detail" in response.json()

    def test_broken_sql_returns_400_with_the_message(self, client: TestClient):
        response = client.post("/sql", json={"sql": "SELECT nonexistent_column FROM dim_slice"})
        assert response.status_code == 400
        assert "query failed" in response.json()["detail"]

    def test_row_limit_is_enforced(self, client: TestClient):
        response = client.post(
            "/sql",
            json={"sql": "SELECT * FROM fact_prediction", "limit": 10},
        )
        payload = response.json()
        assert len(payload["rows"]) == 10
        assert payload["truncated"] is True
        assert payload["row_count"] > 10

    def test_limit_is_capped(self, client: TestClient):
        response = client.post(
            "/sql", json={"sql": "SELECT 1", "limit": 10_000_000}
        )
        assert response.status_code == 422  # out of range for the field

    def test_the_sql_endpoint_cannot_modify_the_warehouse(
    self, db_path: str, client: TestClient
):
        """Belt and braces: even if the guard were bypassed, assert no change."""
        from eval_analytics.warehouse import query

        before = query(db_path, "SELECT COUNT(*) AS n FROM dim_slice").iloc[0]["n"]
        client.post("/sql", json={"sql": "DELETE FROM dim_slice"})
        after = query(db_path, "SELECT COUNT(*) AS n FROM dim_slice").iloc[0]["n"]
        assert before == after
