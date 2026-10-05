"""Star-schema conformance of the built warehouse."""

from __future__ import annotations

from eval_analytics.warehouse import query


def test_all_marts_are_populated(db_path: str):
    counts = query(
        db_path,
        """
        SELECT
          (SELECT COUNT(*) FROM fact_evaluation_run)     AS evaluation_runs,
          (SELECT COUNT(*) FROM fact_prediction)         AS predictions,
          (SELECT COUNT(*) FROM fact_serving_request)    AS serving_requests,
          (SELECT COUNT(*) FROM fact_drift_measurement)  AS drift_measurements
        """,
    ).iloc[0]
    for name, count in counts.items():
        assert int(count) > 0, f"{name} is empty"


def test_no_orphaned_fact_rows(db_path: str):
    row = query(
        db_path,
        """
        SELECT
          (SELECT COUNT(*) FROM fact_prediction p
             LEFT JOIN fact_evaluation_run r USING (evaluation_run_key)
            WHERE r.evaluation_run_key IS NULL) AS orphan_predictions,
          (SELECT COUNT(*) FROM fact_evaluation_run r
             LEFT JOIN dim_model m USING (model_id)
             LEFT JOIN dim_slice s USING (slice_id)
             LEFT JOIN dim_date d USING (date_id)
            WHERE m.model_id IS NULL OR s.slice_id IS NULL OR d.date_id IS NULL)
            AS orphan_runs,
          (SELECT COUNT(*) FROM fact_drift_measurement d
             LEFT JOIN dim_feature f USING (feature_id)
            WHERE f.feature_id IS NULL) AS orphan_drift,
          (SELECT COUNT(*) FROM fact_serving_request s
             LEFT JOIN dim_model m USING (model_id)
            WHERE m.model_id IS NULL) AS orphan_serving
        """,
    ).iloc[0]
    assert sum(int(v) for v in row) == 0, dict(row)


def test_every_slice_is_referenced_by_a_fact(db_path: str):
    """Full conformance: no dimension member is dead weight."""
    orphans = query(
        db_path,
        """
        SELECT s.slice_id, s.slice_type FROM dim_slice s
        WHERE NOT EXISTS (SELECT 1 FROM fact_evaluation_run r WHERE r.slice_id = s.slice_id)
        """,
    )
    assert orphans.empty, f"unreferenced slices: {orphans.to_dict('records')}"


def test_prediction_correctness_agrees_with_its_labels(db_path: str):
    """is_correct must be derivable from the labels, never a separate truth."""
    row = query(
        db_path,
        """
        SELECT COUNT(*) AS n_inconsistent
        FROM fact_prediction
        WHERE is_correct IS NOT NULL
          AND is_correct <> (CASE WHEN true_label = predicted_label THEN 1 ELSE 0 END)
        """,
    ).iloc[0]["n_inconsistent"]
    assert int(row) == 0


def test_class_accuracy_matches_the_predictions_it_summarises(db_path: str):
    """Per-class evaluation rows are derived, so they cannot drift."""
    row = query(
        db_path,
        """
        WITH from_facts AS (
          SELECT r.model_id, r.slice_id,
                 COUNT(*) AS n, SUM(p.is_correct) AS n_correct
          FROM fact_evaluation_run r
          JOIN fact_prediction p ON p.evaluation_run_key = r.evaluation_run_key
          JOIN dim_slice s ON s.slice_id = r.slice_id
          WHERE s.slice_type = 'class'
          GROUP BY r.model_id, r.slice_id
        )
        SELECT COUNT(*) AS n_mismatched
        FROM fact_evaluation_run r
        JOIN dim_slice s ON s.slice_id = r.slice_id
        JOIN from_facts f ON f.model_id = r.model_id AND f.slice_id = r.slice_id
        WHERE s.slice_type = 'class'
          AND (r.n_samples <> f.n OR r.n_correct <> f.n_correct)
        """,
    ).iloc[0]["n_mismatched"]
    assert int(row) == 0


def test_length_bucket_slice_agrees_with_prediction_counts(db_path: str):
    row = query(
        db_path,
        """
        WITH from_facts AS (
          SELECT model_id, doc_length_bucket_id AS slice_id,
                 COUNT(*) AS n, SUM(is_correct) AS n_correct
          FROM fact_prediction
          WHERE doc_length_bucket_id IS NOT NULL AND is_correct IS NOT NULL
          GROUP BY model_id, doc_length_bucket_id
        )
        SELECT COUNT(*) AS n_mismatched
        FROM fact_evaluation_run r
        JOIN dim_slice s ON s.slice_id = r.slice_id
        JOIN from_facts f ON f.model_id = r.model_id AND f.slice_id = r.slice_id
        WHERE s.slice_type = 'length_bucket'
          AND (r.n_samples <> f.n OR r.n_correct <> f.n_correct)
        """,
    ).iloc[0]["n_mismatched"]
    assert int(row) == 0


def test_both_arms_and_multiple_quantisations_are_present(db_path: str):
    arms = query(
        db_path,
        """
        SELECT arm, COUNT(*) AS n FROM dim_model
        GROUP BY arm HAVING COUNT(*) > 0 ORDER BY arm
        """,
    )
    assert {"lora", "full_finetune"} <= set(arms["arm"])

    quants = query(
        db_path, "SELECT DISTINCT quantisation FROM dim_model ORDER BY quantisation"
    )
    assert len(quants) >= 2


def test_probabilities_are_in_range(db_path: str):
    row = query(
        db_path,
        """
        SELECT
          (SELECT COUNT(*) FROM fact_prediction
            WHERE max_probability IS NOT NULL
              AND (max_probability < 0 OR max_probability > 1)) AS bad_max,
          (SELECT COUNT(*) FROM fact_prediction
            WHERE prob_true_label IS NOT NULL
              AND (prob_true_label < 0 OR prob_true_label > 1)) AS bad_true,
          (SELECT COUNT(*) FROM fact_drift_measurement
            WHERE psi IS NOT NULL AND psi < 0) AS bad_psi
        """,
    ).iloc[0]
    assert sum(int(v) for v in row) == 0, dict(row)


def test_slice_projection_view_covers_all_families(db_path: str):
    rows = query(
        db_path,
        """
        SELECT slice_type, COUNT(*) AS n, COUNT(DISTINCT slice_id) AS n_slices
        FROM v_prediction_slice GROUP BY slice_type ORDER BY slice_type
        """,
    )
    types = set(rows["slice_type"])
    assert {"overall", "class", "length_bucket"} <= types
