"""
Unit tests for authentication/auth.py.

Uses a temporary SQLite file per test (via monkeypatching database.db.DB_PATH)
so tests never touch the real healthguardian.db, and each test starts from
a clean schema.
"""
import sqlite3
from pathlib import Path

import pytest

import database.db as db_module
from authentication.auth import (
    EmailTakenError,
    InvalidCredentialsError,
    InvalidInputError,
    UsernameTakenError,
    login,
    signup,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point DB_PATH at a fresh temp file for every test, then init schema."""
    temp_db_path = tmp_path / "test_healthguardian.db"
    monkeypatch.setattr(db_module, "DB_PATH", temp_db_path)
    db_module.init_db()
    yield


def test_signup_creates_user():
    user = signup("alice", "alice@example.com", "password123")
    assert user.username == "alice"
    assert user.email == "alice@example.com"
    assert user.id is not None


def test_signup_rejects_duplicate_username():
    signup("bob", "bob@example.com", "password123")
    with pytest.raises(UsernameTakenError):
        signup("bob", "different@example.com", "password123")


def test_signup_rejects_duplicate_email():
    signup("carol", "carol@example.com", "password123")
    with pytest.raises(EmailTakenError):
        signup("carol2", "carol@example.com", "password123")


def test_signup_rejects_short_password():
    with pytest.raises(InvalidInputError):
        signup("dave", "dave@example.com", "short1")


def test_signup_rejects_invalid_email():
    with pytest.raises(InvalidInputError):
        signup("eve", "not-an-email", "password123")


def test_login_succeeds_with_correct_credentials():
    signup("frank", "frank@example.com", "password123")
    user = login("frank", "password123")
    assert user.username == "frank"


def test_login_fails_with_wrong_password():
    signup("grace", "grace@example.com", "password123")
    with pytest.raises(InvalidCredentialsError):
        login("grace", "wrongpassword")


def test_login_fails_with_unknown_username():
    with pytest.raises(InvalidCredentialsError):
        login("nobody", "password123")


def test_password_is_hashed_not_stored_plaintext():
    signup("henry", "henry@example.com", "password123")
    conn = db_module.get_connection()
    row = conn.execute("SELECT password_hash FROM users WHERE username = ?", ("henry",)).fetchone()
    conn.close()
    assert row["password_hash"] != "password123"
    assert row["password_hash"].startswith("$2b$")  # bcrypt hash prefix
