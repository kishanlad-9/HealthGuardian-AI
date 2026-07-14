"""
Session-state helpers.

Kept separate from auth.py deliberately: auth.py has zero Streamlit
dependency (it's pure Python + SQLite + bcrypt), which makes it directly
unit-testable without mocking Streamlit. This module is the thin
Streamlit-specific glue on top.
"""
import streamlit as st

from authentication.auth import User

SESSION_KEY = "current_user"


def get_current_user() -> User | None:
    return st.session_state.get(SESSION_KEY)


def is_logged_in() -> bool:
    return get_current_user() is not None


def log_in_session(user: User) -> None:
    st.session_state[SESSION_KEY] = user


def log_out_session() -> None:
    st.session_state.pop(SESSION_KEY, None)


def require_login() -> User:
    """Call at the top of any page that needs a logged-in user. Stops
    page execution with a friendly message if nobody is logged in, instead
    of letting the page crash on a None user further down."""
    user = get_current_user()
    if user is None:
        st.warning("Please log in to view this page.")
        st.stop()
    return user
