-- =========================================================
-- PREDICTION MONITORING
-- =========================================================

CREATE SCHEMA IF NOT EXISTS monitoring;


-- =========================================================
-- PREDICTION LOG TABLE
-- =========================================================

CREATE TABLE IF NOT EXISTS monitoring.prediction_logs
(
    prediction_id BIGSERIAL PRIMARY KEY,

    model_name VARCHAR(100) NOT NULL,

    model_version VARCHAR(50) NOT NULL,

    endpoint VARCHAR(150),

    predicted_class VARCHAR(100),

    probability NUMERIC(10, 6),

    threshold NUMERIC(10, 6),

    feature_snapshot JSONB,

    explainability JSONB,

    latency_ms NUMERIC(12, 3),

    status VARCHAR(30) NOT NULL DEFAULT 'success',

    error_message TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =========================================================
-- INDEXES
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_prediction_logs_model
ON monitoring.prediction_logs(model_name);


CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at
ON monitoring.prediction_logs(created_at);


CREATE INDEX IF NOT EXISTS idx_prediction_logs_status
ON monitoring.prediction_logs(status);