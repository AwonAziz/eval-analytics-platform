-- V002__fact_tables.sql
-- The four fact tables. Each one is declared at an explicit grain with a
-- CHECK-constrained natural key so that a bad load fails loudly instead of
-- silently producing double-counted analytics.

-- Grain: one row per (experiment, model, slice, date, window).
-- The window is part of the grain because a single run evaluates many
-- windows that can share a date and a regime label (two 'baseline' windows
-- in one day), so (experiment, model, slice, date) alone is not unique.
CREATE TABLE IF NOT EXISTS fact_evaluation_run (
    evaluation_run_key  BIGINT PRIMARY KEY,
    experiment_id       VARCHAR NOT NULL REFERENCES dim_experiment(experiment_id),
    model_id            VARCHAR NOT NULL REFERENCES dim_model(model_id),
    slice_id            VARCHAR NOT NULL REFERENCES dim_slice(slice_id),
    date_id             DATE    NOT NULL REFERENCES dim_date(date_id),
    window_id           VARCHAR,
    n_samples           INTEGER NOT NULL CHECK (n_samples > 0),
    n_correct           INTEGER CHECK (n_correct IS NULL OR n_correct BETWEEN 0 AND n_samples),
    accuracy            DOUBLE  NOT NULL CHECK (accuracy BETWEEN 0 AND 1),
    accuracy_ci_low     DOUBLE,
    accuracy_ci_high    DOUBLE,
    macro_f1            DOUBLE CHECK (macro_f1 IS NULL OR macro_f1 BETWEEN 0 AND 1),
    ece                 DOUBLE CHECK (ece IS NULL OR ece >= 0),
    mce                 DOUBLE CHECK (mce IS NULL OR mce >= 0),
    brier               DOUBLE CHECK (brier IS NULL OR brier BETWEEN 0 AND 1),
    trainable_params    BIGINT,
    seed                BIGINT,
    evaluation_seconds  DOUBLE,
    source_system       VARCHAR NOT NULL,
    data_origin         VARCHAR NOT NULL CHECK (data_origin IN ('real', 'synthetic')),
    CONSTRAINT uq_fact_evaluation_run
        UNIQUE (experiment_id, model_id, slice_id, date_id, window_id)
);

-- Grain: one row per (evaluation run, test example).
CREATE TABLE IF NOT EXISTS fact_prediction (
    prediction_key       BIGINT PRIMARY KEY,
    evaluation_run_key   BIGINT    NOT NULL REFERENCES fact_evaluation_run(evaluation_run_key),
    experiment_id        VARCHAR   NOT NULL,
    model_id             VARCHAR   NOT NULL,
    date_id              DATE      NOT NULL,
    example_id           VARCHAR   NOT NULL,
    window_id            VARCHAR,
    true_label           VARCHAR,
    predicted_label      VARCHAR,
    prob_true_label      DOUBLE CHECK (prob_true_label IS NULL OR prob_true_label BETWEEN 0 AND 1),
    prob_predicted_label DOUBLE CHECK (prob_predicted_label IS NULL OR prob_predicted_label BETWEEN 0 AND 1),
    max_probability      DOUBLE CHECK (max_probability IS NULL OR max_probability BETWEEN 0 AND 1),
    confidence_margin    DOUBLE,
    entropy              DOUBLE CHECK (entropy IS NULL OR entropy >= 0),
    is_correct           INTEGER CHECK (is_correct IS NULL OR is_correct IN (0, 1)),
    is_abstained         BOOLEAN,
    n_tokens             INTEGER CHECK (n_tokens IS NULL OR n_tokens > 0),
    doc_length_bucket_id VARCHAR REFERENCES dim_slice(slice_id),
    data_origin          VARCHAR   NOT NULL CHECK (data_origin IN ('real', 'synthetic')),
    CONSTRAINT uq_fact_prediction UNIQUE (evaluation_run_key, example_id)
);

-- Grain: one row per timed serving request under one quantisation config.
CREATE TABLE IF NOT EXISTS fact_serving_request (
    serving_request_key BIGINT PRIMARY KEY,
    model_id             VARCHAR NOT NULL REFERENCES dim_model(model_id),
    date_id              DATE    NOT NULL REFERENCES dim_date(date_id),
    slice_id             VARCHAR REFERENCES dim_slice(slice_id),
    quantisation         VARCHAR NOT NULL,
    batch_size           INTEGER CHECK (batch_size IS NULL OR batch_size > 0),
    input_tokens         INTEGER CHECK (input_tokens IS NULL OR input_tokens > 0),
    latency_ms           DOUBLE   NOT NULL CHECK (latency_ms >= 0),
    model_size_bytes     BIGINT   CHECK (model_size_bytes IS NULL OR model_size_bytes > 0),
    peak_memory_mb       DOUBLE   CHECK (peak_memory_mb IS NULL OR peak_memory_mb >= 0),
    throughput_tps       DOUBLE   CHECK (throughput_tps IS NULL OR throughput_tps >= 0),
    request_index        INTEGER,
    data_origin          VARCHAR  NOT NULL CHECK (data_origin IN ('real', 'synthetic')),
    CONSTRAINT uq_fact_serving_request
        UNIQUE (model_id, quantisation, batch_size, input_tokens, request_index)
);

-- Grain: one row per (feature, window, metric name).
CREATE TABLE IF NOT EXISTS fact_drift_measurement (
    drift_measurement_key BIGINT PRIMARY KEY,
    feature_id            VARCHAR NOT NULL REFERENCES dim_feature(feature_id),
    model_id              VARCHAR REFERENCES dim_model(model_id),
    date_id               DATE    NOT NULL REFERENCES dim_date(date_id),
    slice_id              VARCHAR REFERENCES dim_slice(slice_id),
    window_id             VARCHAR,
    measurement_ts        TIMESTAMP NOT NULL,
    metric_name           VARCHAR   NOT NULL,
    metric_value          DOUBLE,
    -- Denormalised short-hands, populated when the metric is one of the
    -- three standard drift statistics. Keeps "PSI/KS/JS per feature over
    -- time" a single pivot rather than a wide table.
    psi                  DOUBLE,
    ks_statistic         DOUBLE CHECK (ks_statistic IS NULL OR ks_statistic BETWEEN 0 AND 1),
    ks_pvalue            DOUBLE CHECK (ks_pvalue IS NULL OR ks_pvalue BETWEEN 0 AND 1),
    js_divergence        DOUBLE CHECK (js_divergence IS NULL OR js_divergence BETWEEN 0 AND 1),
    baseline_value       DOUBLE,
    current_value        DOUBLE,
    sample_size          INTEGER,
    threshold_moderate   DOUBLE,
    threshold_severe     DOUBLE,
    severity             VARCHAR,
    data_origin          VARCHAR NOT NULL CHECK (data_origin IN ('real', 'synthetic')),
    CONSTRAINT uq_fact_drift_measurement
        UNIQUE (feature_id, window_id, metric_name)
);