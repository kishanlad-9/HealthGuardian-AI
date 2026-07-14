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
