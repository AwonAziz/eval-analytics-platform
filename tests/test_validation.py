"""Ingestion validation: contracts reject bad rows and rejections are counted."""

from __future__ import annotations

import json

import pytest

from eval_analytics.ingest.contracts import CONTRACTS
from eval_analytics.ingest.validate import validate_rows


def _traffic(**overrides):
    row = {
        "request_id": "req_1",
        "ts": "2026-10-01T12:00:00+00:00",
        "window_id": "w1",
        "run_id": "run_1",
        "gold_intent": "card_arrival",
        "pred_intent": "card_arrival",
        "confidence": 0.93,
        "abstained": False,
        "in_scope": True,
        "latency_ms": 12.5,
        "text_length_tokens": 14,
    }
    row.update(overrides)
    return row


def _eval_run(**overrides):
    row = {
        "experiment_id": "exp-1",
        "model_id": "m1",
        "base_model": "distilbert",
        "arm": "lora",
        "method": "lora",
        "quantisation": "fp32",
        "adapter_rank": 16,
        "n_samples": 1000,
        "n_correct": 900,
        "accuracy": 0.9,
        "accuracy_ci_low": 0.88,
        "accuracy_ci_high": 0.92,
        "macro_f1": 0.89,
        "ece": 0.018,
        "mce": 0.12,
        "brier": 0.1,
        "temperature": 0.45,
        "trainable_params": 887808,
        "total_params": 66_000_000,
        "seed": 1,
        "window_id": None,
        "started_at": "2026-10-03T09:15:00+00:00",
        "finished_at": "2026-10-03T09:56:00+00:00",
    }
    row.update(overrides)
    return row


class TestContractsExist:
    def test_every_dataset_has_a_contract(self):
        for name in ("traffic", "metric", "window", "eval_run", "prediction",
                     "serving", "drift", "model_card"):
            assert name in CONTRACTS

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="no contract registered"):
            validate_rows("not_a_dataset", [{}], "test")


class TestAcceptedRows:
    def test_well_formed_traffic_is_accepted(self):
        result = validate_rows("traffic", [_traffic()], "test")
        assert result.rows_rejected == 0
        assert len(result.accepted) == 1

    def test_unlabelled_traffic_keeps_a_null_label(self):
        """Unlabeled production traffic is kept, not dropped or invented."""
        result = validate_rows("traffic", [_traffic(gold_intent=None)], "test")
        assert result.rows_rejected == 0
        assert result.accepted[0]["gold_intent"] is None

    def test_non_finetune_arm_is_allowed(self):
        result = validate_rows(
            "eval_run", [_eval_run(arm="none", method="calibrated_logreg")], "test"
        )
        assert result.rows_rejected == 0


class TestRejectedRows:
    @pytest.mark.parametrize(
        "overrides, reason",
        [
            ({"confidence": 1.4}, "confidence above 1"),
            ({"confidence": -0.1}, "negative confidence"),
            ({"latency_ms": -5.0}, "negative latency"),
            ({"request_id": ""}, "empty request id"),
            ({"ts": "not-a-timestamp"}, "unparseable timestamp"),
        ],
    )
    def test_malformed_traffic_is_rejected(self, overrides, reason):
        result = validate_rows("traffic", [_traffic(**overrides)], "test")
        assert result.rows_rejected == 1, reason
        assert result.accepted == []

    def test_n_correct_cannot_exceed_n_samples(self):
        result = validate_rows(
            "eval_run", [_eval_run(n_correct=1200, n_samples=1000)], "test"
        )
        assert result.rows_rejected == 1
        assert "exceeds n_samples" in " ".join(result.rejected[0].error_messages)

    def test_accuracy_outside_its_own_confidence_interval_is_rejected(self):
        result = validate_rows(
            "eval_run",
            [_eval_run(accuracy=0.70, accuracy_ci_low=0.88, accuracy_ci_high=0.92)],
            "test",
        )
        assert result.rows_rejected == 1
        assert "outside CI" in " ".join(result.rejected[0].error_messages)

    def test_inverted_confidence_interval_is_rejected(self):
        result = validate_rows(
            "eval_run", [_eval_run(accuracy_ci_low=0.95, accuracy_ci_high=0.85)], "test"
        )
        assert result.rows_rejected == 1

    def test_unknown_field_is_rejected(self):
        """A renamed upstream column must not pass silently."""
        result = validate_rows("traffic", [_traffic(surprise_field=1)], "test")
        assert result.rows_rejected == 1
        assert "extra_forbidden" in result.rejected[0].error_codes

    def test_prob_true_label_without_a_true_label_is_rejected(self):
        row = {
            "experiment_id": "e", "model_id": "m", "example_id": "x",
            "true_label": None, "predicted_label": "a",
            "prob_true_label": 0.5, "prob_predicted_label": 0.6,
            "max_probability": 0.6, "confidence_margin": 0.1, "entropy": 0.5,
            "n_tokens": 12, "window_id": None,
        }
        result = validate_rows("prediction", [row], "test")
        assert result.rows_rejected == 1

    def test_trainable_params_cannot_exceed_total(self):
        row = {
            "model_id": "m", "base_model": "b", "arm": "lora", "method": "lora",
            "adapter_rank": 16, "trainable_params": 10, "total_params": 5,
            "trainable_pct": 200.0, "quantisation": "fp32", "dataset": "d",
            "created_at": None, "is_champion": False,
        }
        result = validate_rows("model_card", [row], "test")
        assert result.rows_rejected == 1


class TestCounting:
    def test_good_and_bad_rows_are_counted_separately(self):
        rows = [_traffic(request_id=f"ok{i}") for i in range(4)]
        rows.append(_traffic(request_id="bad", confidence=5.0))
        rows.append(_traffic(request_id="bad2", ts="nope"))
        result = validate_rows("traffic", rows, "test")

        assert result.rows_in == 6
        assert len(result.accepted) == 4
        assert result.rows_rejected == 2
        assert result.rejection_rate == pytest.approx(2 / 6)

    def test_rejections_carry_codes_and_the_raw_payload(self):
        result = validate_rows("traffic", [_traffic(confidence=2.0)], "telemetry_db")
        rejection = result.rejected[0]
        assert rejection.source_name == "telemetry_db"
        assert rejection.source_row_id == "req_1"
        assert rejection.error_codes
        assert json.loads(rejection.raw_payload)["confidence"] == 2.0

    def test_error_code_counts_are_aggregated(self):
        rows = [
            _traffic(request_id="a", confidence=5.0),
            _traffic(request_id="b", confidence=-1.0),
        ]
        result = validate_rows("traffic", rows, "test")
        counts = result.error_code_counts()
        assert sum(counts.values()) >= 2


class TestRejectionsReachTheWarehouse:
    def test_rejected_rows_are_recorded_in_the_audit_table(self, db_path: str):
        from eval_analytics.warehouse import query

        # The real build rejects nothing; assert the table exists and agrees.
        row = query(
            db_path,
            """
            SELECT
              (SELECT COUNT(*) FROM ingestion_rejection) AS n_rejections,
              (SELECT COUNT(*) FROM ingestion_run WHERE status = 'success') AS n_success
            """,
        ).iloc[0]
        assert int(row["n_rejections"]) == 0
        assert int(row["n_success"]) >= 1

    def test_a_build_with_corrupt_artifacts_records_rejections(self, tmp_path, repo_root):
        """End-to-end: a malformed row is counted, not silently dropped."""
        from eval_analytics.config import Settings
        from eval_analytics.ingest.artifacts import generate_all
        from eval_analytics.ingest.pipeline import run_pipeline
        from eval_analytics.warehouse import query, reset_engines

        generated = tmp_path / "raw" / "generated"
        generate_all(generated)

        # Corrupt one artifact report so exactly one row fails validation.
        target = generated / "exp-001-lora" / "eval_report.json"
        report = json.loads(target.read_text(encoding="utf-8"))
        report["accuracy"] = 1.9  # physically impossible
        target.write_text(json.dumps(report), encoding="utf-8")

        settings = Settings(
            duckdb_path=tmp_path / "corrupt.duckdb",
            raw_dir=tmp_path / "raw",
            generated_dir=generated,
            telemetry_db_path=None,
        )
        result = run_pipeline(
            db_path=settings.duckdb_path,
            settings=settings,
            regenerate_artifacts=False,
            fail_fast=False,
            verbose=False,
        )
        assert result.rows_rejected == 1
        assert result.rows_accepted == result.rows_landed - 1

        rejections = query(
            settings.duckdb_path,
            """
            SELECT dataset_name, source_name, source_row_id, error_codes
            FROM ingestion_rejection
            """,
        )
        assert len(rejections) == 1
        assert rejections.iloc[0]["dataset_name"] == "eval_run"
        assert "less_than_equal" in rejections.iloc[0]["error_codes"]

        # And the bad row did not reach the mart, while the good ones did.
        n_runs = query(
            settings.duckdb_path,
            "SELECT COUNT(*) AS n FROM fact_evaluation_run",
        ).iloc[0]["n"]
        assert int(n_runs) > 0
        reset_engines()
