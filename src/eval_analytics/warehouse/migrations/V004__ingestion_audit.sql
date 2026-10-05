-- V004__ingestion_audit.sql
-- Pipeline observability. `ingestion_rejection` is the counted-rejection
-- table: one row per rejected source row, with the machine-readable error
-- codes that Pydantic produced. `mart_freshness` is what CI asserts against.

CREATE TABLE IF NOT EXISTS ingestion_run (
    pipeline_run_id   VARCHAR PRIMARY KEY,
    started_at        TIMESTAMP NOT NULL,
    finished_at       TIMESTAMP,
    status            VARCHAR NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    source_name       VARCHAR NOT NULL,
    data_origin       VARCHAR NOT NULL CHECK (data_origin IN ('real', 'synthetic')),
    rows_landed       BIGINT NOT NULL DEFAULT 0,
    rows_accepted     BIGINT NOT NULL DEFAULT 0,
    rows_rejected     BIGINT NOT NULL DEFAULT 0,
    error             VARCHAR
);

CREATE TABLE IF NOT EXISTS ingestion_rejection (
    rejection_id      BIGINT PRIMARY KEY,
    pipeline_run_id   VARCHAR NOT NULL REFERENCES ingestion_run(pipeline_run_id),
    source_name       VARCHAR NOT NULL,
    dataset_name      VARCHAR NOT NULL,
    source_row_id     VARCHAR,
    error_count       INTEGER NOT NULL,
    error_codes       VARCHAR NOT NULL,
    error_messages    VARCHAR NOT NULL,
    raw_payload       VARCHAR,
    rejected_at       TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS mart_freshness (
    mart_name         VARCHAR PRIMARY KEY,
    row_count         BIGINT NOT NULL,
    max_date_id       DATE,
    max_loaded_at     TIMESTAMP NOT NULL,
    source_max_ts     TIMESTAMP,
    source_row_count  BIGINT,
    is_stale          BOOLEAN NOT NULL,
    staleness_reason  VARCHAR
);

-- Convenience views used by the API and dashboard.

-- Headline model quality on the 'overall' slice only.
CREATE OR REPLACE VIEW v_model_quality_headline AS
SELECT
    r.experiment_id,
    m.model_id,
    m.arm,
    m.quantisation,
    m.trainable_params,
    r.n_samples,
    r.accuracy,
    r.accuracy_ci_low,
    r.accuracy_ci_high,
    r.macro_f1,
    r.ece,
    r.mce,
    r.brier,
    r.date_id,
    r.data_origin
FROM fact_evaluation_run r
JOIN dim_model m  ON m.model_id = r.model_id
JOIN dim_slice s ON s.slice_id = r.slice_id
WHERE s.slice_type = 'overall';

-- One row per (feature, window) with PSI/KS/JS pivoted side by side.
-- Grouping on the natural grain, not the surrogate key, so the three
-- statistics for a feature-window collapse into one wide row.
CREATE OR REPLACE VIEW v_drift_pivot AS
SELECT
    d.feature_id,
    f.feature_type,
    d.window_id,
    d.measurement_ts,
    d.date_id,
    MAX(CASE WHEN d.metric_name = 'psi'           THEN d.metric_value END) AS psi,
    MAX(CASE WHEN d.metric_name = 'ks_statistic'  THEN d.metric_value END) AS ks_statistic,
    MAX(CASE WHEN d.metric_name = 'js_divergence' THEN d.metric_value END) AS js_divergence,
    MAX(d.severity)       AS severity,
    MAX(d.sample_size)    AS sample_size,
    MAX(d.baseline_value) AS baseline_value,
    MAX(d.current_value)  AS current_value,
    ANY_VALUE(d.data_origin) AS data_origin
FROM fact_drift_measurement d
JOIN dim_feature f ON f.feature_id = d.feature_id
GROUP BY d.feature_id, f.feature_type, d.window_id, d.measurement_ts, d.date_id;