"""
Database connection helper.

Design decision: a single `get_connection()` function, called fresh wherever
a query is needed, rather than one long-lived global connection. SQLite
connections aren't safely shared across Streamlit's rerun model (each
interaction can run in a different thread), so "open, query, close" per
call is the correct pattern here - not a premature optimization to avoid.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "healthguardian.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with foreign keys enabled and Row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables from schema.sql if they don't already exist.

    Safe to call on every app startup - CREATE TABLE IF NOT EXISTS is
    idempotent.
    """
    schema_sql = SCHEMA_PATH.read_text()
    conn = get_connection()
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()
