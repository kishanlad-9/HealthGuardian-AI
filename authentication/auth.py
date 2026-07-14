"""
Authentication logic: signup and login against the SQLite users table.

Design decisions:
- Custom exceptions (not bare bool returns) so the Streamlit page can show
  specific, actionable error messages without string-matching.
- bcrypt for hashing: includes salt generation automatically, and its
  deliberately-slow design resists brute-force attacks better than a fast
  hash (e.g. sha256) would - the right tradeoff for password storage
  specifically, even though it's slower than we'd want elsewhere.
- No password_hash ever leaves this module - callers get a User dataclass
  without it, so it can't accidentally end up in a log line or UI.
"""
import sqlite3
from dataclasses import dataclass

import bcrypt

from database.db import get_connection
from utils.validators import is_valid_email, is_valid_password, is_valid_username


class AuthError(Exception):
    """Base class for authentication errors - lets Streamlit code catch
    broadly (`except AuthError`) or specifically, as needed."""


class UsernameTakenError(AuthError):
    pass


class EmailTakenError(AuthError):
    pass


class InvalidInputError(AuthError):
    pass


class InvalidCredentialsError(AuthError):
    """Deliberately generic - never reveals whether it was the username or
    the password that was wrong, which would leak whether an account exists."""


@dataclass(frozen=True)
class User:
    id: int
    username: str
    email: str


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def signup(username: str, email: str, password: str) -> User:
    """Create a new user. Raises InvalidInputError / UsernameTakenError /
    EmailTakenError on failure; returns the created User on success."""
    username = username.strip()
    email = email.strip().lower()

    valid_username, username_error = is_valid_username(username)
    if not valid_username:
        raise InvalidInputError(username_error)

    if not is_valid_email(email):
        raise InvalidInputError("Please enter a valid email address.")

    valid_password, password_error = is_valid_password(password)
    if not valid_password:
        raise InvalidInputError(password_error)

    password_hash = _hash_password(password)

    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
        return User(id=cursor.lastrowid, username=username, email=email)
    except sqlite3.IntegrityError as e:
        # UNIQUE constraint violation - figure out which column, for a
        # helpful error message, by checking what's already taken.
        existing = conn.execute(
            "SELECT username, email FROM users WHERE username = ? OR email = ?",
            (username, email),
        ).fetchone()
        if existing and existing["username"] == username:
            raise UsernameTakenError("That username is already taken.") from e
        if existing and existing["email"] == email:
            raise EmailTakenError("An account with that email already exists.") from e
        raise InvalidInputError("Could not create account.") from e
    finally:
        conn.close()


def login(username: str, password: str) -> User:
    """Verify credentials. Raises InvalidCredentialsError on any failure
    (unknown username OR wrong password - see InvalidCredentialsError docstring)."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    finally:
        conn.close()

    if row is None or not _verify_password(password, row["password_hash"]):
        raise InvalidCredentialsError("Incorrect username or password.")

    return User(id=row["id"], username=row["username"], email=row["email"])
