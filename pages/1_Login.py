"""Login / signup page - split-panel layout (form left, illustration
right), matching the reference style the user provided. Appears
automatically in Streamlit's sidebar (pages/ convention)."""
import streamlit as st

from authentication.auth import AuthError, login, signup
from authentication.session import (
    get_current_user,
    is_logged_in,
    log_in_session,
    log_out_session,
)
from database.db import init_db
from utils.illustrations import LOGIN_ILLUSTRATION_SVG
from utils.theme import inject_theme

st.set_page_config(page_title="Login - HealthGuardian AI", page_icon="🔐", layout="wide")
inject_theme()
init_db()


def render_logged_in_view() -> None:
    user = get_current_user()
    st.title("🔐 Account")
    st.success(f"Logged in as **{user.username}** ({user.email})")
    if st.button("Log out"):
        log_out_session()
        st.rerun()


def render_login_form() -> None:
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

    if submitted:
        try:
            user = login(username, password)
            log_in_session(user)
            st.rerun()
        except AuthError as e:
            st.error(str(e))


def render_signup_form() -> None:
    with st.form("signup_form"):
        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)

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
    col_form, col_illustration = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<span class="hg-pill">Welcome</span>', unsafe_allow_html=True)
        st.markdown("## Login to Your Account")
        st.markdown(
            '<p style="color: var(--text-body); margin-top: -0.6rem;">'
            "Sign in to view your prediction history, or create an account to get started."
            "</p>",
            unsafe_allow_html=True,
        )

        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
        with login_tab:
            render_login_form()
        with signup_tab:
            render_signup_form()

    with col_illustration:
        st.markdown(LOGIN_ILLUSTRATION_SVG, unsafe_allow_html=True)


if is_logged_in():
    render_logged_in_view()
else:
    render_logged_out_view()
