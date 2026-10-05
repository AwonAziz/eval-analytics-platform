-- V001__star_dimensions.sql
-- Conformed dimensions for the evaluation + telemetry warehouse.
--
-- Dimensions use natural (surrogate-free) primary keys so that joins stay
-- readable in the SQL API and in the dashboard. Fact tables carry a
-- deterministic integer surrogate key derived from their natural grain by
-- eval_analytics.warehouse.keys.stable_key().

CREATE TABLE IF NOT EXISTS dim_model (
    model_id          VARCHAR PRIMARY KEY,
    base_model        VARCHAR    NOT NULL,
    arm               VARCHAR    NOT NULL CHECK (arm IN ('lora', 'full_finetune', 'none')),
    method            VARCHAR    NOT NULL,
    quantisation      VARCHAR    NOT NULL DEFAULT 'fp32',
    adapter_rank      INTEGER,
    trainable_params  BIGINT,
    total_params      BIGINT,
    trainable_pct     DOUBLE,
    source_dataset    VARCHAR,
    created_at        TIMESTAMP,
    is_champion       BOOLEAN    DEFAULT FALSE,
    data_origin       VARCHAR    NOT NULL DEFAULT 'real'
);

CREATE TABLE IF NOT EXISTS dim_experiment (
    experiment_id  VARCHAR PRIMARY KEY,
    name           VARCHAR NOT NULL,
    hypothesis     VARCHAR,
    owner          VARCHAR,
    started_at     TIMESTAMP,
    ended_at       TIMESTAMP,
    status         VARCHAR,
    data_source    VARCHAR,
    tags           VARCHAR,
    data_origin    VARCHAR NOT NULL DEFAULT 'real'
);

CREATE TABLE IF NOT EXISTS dim_feature (
    feature_id        VARCHAR PRIMARY KEY,
    feature_type      VARCHAR NOT NULL CHECK (feature_type IN ('numeric', 'categorical')),
    source_system     VARCHAR NOT NULL,
    is_drift_tracked  BOOLEAN DEFAULT TRUE,
    description       VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_slice (
    slice_id       VARCHAR PRIMARY KEY,
    slice_type     VARCHAR NOT NULL,
    slice_name     VARCHAR NOT NULL,
    ordinal        INTEGER,
    definition     VARCHAR,
    source_system  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id      DATE PRIMARY KEY,
    year         INTEGER NOT NULL,
    quarter      INTEGER NOT NULL,
    month        INTEGER NOT NULL,
    week         INTEGER NOT NULL,
    day_of_week  INTEGER NOT NULL,
    is_weekend   BOOLEAN NOT NULL
);