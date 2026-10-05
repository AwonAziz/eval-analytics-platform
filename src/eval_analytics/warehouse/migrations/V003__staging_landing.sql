-- V003__staging_landing.sql
-- Landing zone. Every column is nullable text/JSON so that a malformed
-- upstream artifact lands intact and is rejected by the validation layer
-- with a recorded reason, rather than blowing up the load mid-cast.

CREATE TABLE IF NOT EXISTS stg_traffic_raw (
    _ingest_run_id    VARCHAR,
    _source_name      VARCHAR,
    _source_row_id    VARCHAR,
    payload           VARCHAR NOT NULL,
    landed_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stg_metrics_raw (
    _ingest_run_id    VARCHAR,
    _source_name      VARCHAR,
    _source_row_id    VARCHAR,
    payload           VARCHAR NOT NULL,
    landed_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stg_windows_raw (
    _ingest_run_id    VARCHAR,
    _source_name      VARCHAR,
    _source_row_id    VARCHAR,
    payload           VARCHAR NOT NULL,
    landed_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stg_eval_run_raw (
    _ingest_run_id    VARCHAR,
    _source_name      VARCHAR,
    _source_row_id    VARCHAR,
    payload           VARCHAR NOT NULL,
    landed_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stg_predictions_raw (
    _ingest_run_id    VARCHAR,
    _source_name      VARCHAR,
    _source_row_id    VARCHAR,
    payload           VARCHAR NOT NULL,
    landed_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stg_serving_raw (
    _ingest_run_id    VARCHAR,
    _source_name      VARCHAR,
    _source_row_id    VARCHAR,
    payload           VARCHAR NOT NULL,
    landed_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stg_drift_raw (
    _ingest_run_id    VARCHAR,
    _source_name      VARCHAR,
    _source_row_id    VARCHAR,
    payload           VARCHAR NOT NULL,
    landed_at         TIMESTAMP NOT NULL
);