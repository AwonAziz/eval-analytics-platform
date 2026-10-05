-- V005__prediction_slice_projection.sql
-- Additive migration: denormalise the drift-regime slice onto
-- fact_prediction, then project every prediction onto all three slice
-- families it belongs to.
--
-- Why a column rather than a join at query time: a prediction belongs to its
-- class slice, its length-bucket slice *and* its drift-regime slice at once,
-- but the window's regime label lives only in the upstream windows table and
-- is not carried in the fact. Storing the resolved slice id keeps the star
-- schema conforming without dragging a window dimension into every join.
--
-- v_prediction_slice then answers "how did each model do on every slice"
-- with one pass instead of a UNION per slice family.

-- No REFERENCES clause: DuckDB cannot add a column with a foreign key
-- constraint. Conformance for this column is enforced by the loader (the
-- regime id is only ever written from a registered dim_slice member) and by
-- `eap check`, which asserts that no fact key is orphaned.
ALTER TABLE fact_prediction ADD COLUMN IF NOT EXISTS regime_slice_id VARCHAR;

CREATE OR REPLACE VIEW v_prediction_slice AS
SELECT
    p.prediction_key,
    p.example_id,
    p.model_id,
    m.arm,
    m.quantisation,
    m.base_model,
    m.trainable_params,
    p.experiment_id,
    p.date_id,
    p.window_id,
    p.true_label,
    p.predicted_label,
    p.is_correct,
    p.max_probability,
    p.prob_true_label,
    p.confidence_margin,
    p.entropy,
    p.n_tokens,
    p.data_origin,
    s.slice_id,
    s.slice_type,
    s.slice_name,
    s.ordinal
FROM fact_prediction p
JOIN dim_model m ON m.model_id = p.model_id
JOIN dim_slice s
  ON s.slice_id = 'overall'
  OR s.slice_id = p.doc_length_bucket_id
  OR s.slice_id = p.regime_slice_id
  OR (p.true_label IS NOT NULL AND s.slice_id = 'class_' || p.true_label);