"""Login / signup page - appears automatically in Streamlit's sidebar
because it lives under pages/ (Streamlit multi-page-app convention)."""
import streamlit as st

from authentication.auth import AuthError, login, signup
from authentication.session import (
    get_current_user,
    is_logged_in,
    log_in_session,
    log_out_session,
)
from database.db import init_db

st.set_page_config(page_title="Login - HealthGuardian AI", page_icon="🔐")
init_db()  # idempotent - safe to call on every page load


def render_logged_in_view() -> None:
    user = get_current_user()
    st.title("🔐 Account")
    st.success(f"Logged in as **{user.username}** ({user.email})")
    if st.button("Log out"):
        log_out_session()
        st.rerun()


def render_login_form() -> None:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            user = login(username, password)
            log_in_session(user)
            st.rerun()
        except AuthError as e:
            st.error(str(e))


def render_signup_form() -> None:
    with st.form("signup_form"):
        username = st.text_input("Choose a username")
        email = st.text_input("Email")
        password = st.text_input("Choose a password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        if password != confirm_password:
            st.error("Passwords do not match.")
            return
        try:
            user = signup(username, email, password)
            log_in_session(user)
            st.success("Account created!")
            st.rerun()
        except AuthError as e:
            st.error(str(e))


def render_logged_out_view() -> None:
    st.title("🔐 Log in or sign up")
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
    with login_tab:
        render_login_form()
    with signup_tab:
        render_signup_form()


if is_logged_in():
    render_logged_in_view()
else:
    render_logged_out_view()
