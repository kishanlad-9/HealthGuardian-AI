-- Milestone 2: users table only, enough to support authentication.
-- Milestone 3 will add prediction_history and any supporting tables,
-- and this file will grow to reflect the full schema.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Milestone 3: prediction_history.
--
-- Design decision: one generic table for ALL disease types (heart, diabetes,
-- and future CKD/liver), rather than a separate table per disease. Each
-- disease has a different feature set (heart disease uses cholesterol/chest
-- pain type; diabetes uses glucose/BMI/insulin), so those go in
-- `input_features` as JSON rather than as fixed columns - adding a new
-- disease in Milestone 7+ then requires zero schema migration, just a new
-- `disease_type` value. The tradeoff: can't easily query "average glucose
-- across all predictions" with plain SQL (would need JSON functions) - an
-- acceptable cost for a portfolio app that isn't doing that kind of analytics.
--
-- shap_values is nullable because it's populated starting in Milestone 6 -
-- earlier predictions (or if SHAP computation fails) simply have NULL here.
CREATE TABLE IF NOT EXISTS prediction_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL,
    disease_type     TEXT NOT NULL CHECK (disease_type IN ('heart_disease', 'diabetes')),
    input_features   TEXT NOT NULL,   -- JSON: {"age": 54, "cholesterol": 240, ...}
    risk_probability REAL NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    risk_label       TEXT NOT NULL CHECK (risk_label IN ('Low', 'Medium', 'High')),
    model_version     TEXT NOT NULL, -- e.g. "heart_xgboost_v1" - which saved_models/ artifact produced this
    shap_values       TEXT,           -- JSON, nullable until Milestone 6
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Every history lookup filters by user_id (and often disease_type too),
-- so both are indexed. created_at is indexed for "most recent N" queries
-- on the dashboard.
CREATE INDEX IF NOT EXISTS idx_prediction_history_user_id
    ON prediction_history(user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_history_disease_type
    ON prediction_history(disease_type);
CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at
    ON prediction_history(created_at);
