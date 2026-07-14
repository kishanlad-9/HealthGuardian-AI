"""
HealthGuardian AI - main Streamlit entry point.

This file stays intentionally thin: it configures the page and renders a
landing view. Actual feature pages live under pages/ (Streamlit's
multi-page-app convention - any .py file there becomes a sidebar page
automatically). Business logic belongs in authentication/, ml/, database/,
and utils/ - not here - so this file remains easy to read as the app grows.
"""
import streamlit as st

from authentication.session import get_current_user, is_logged_in
from database.db import init_db

st.set_page_config(
    page_title="HealthGuardian AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()  # idempotent - creates tables on first run, no-ops after


def render_landing() -> None:
    """Render the landing page."""
    st.title("🩺 HealthGuardian AI")
    st.subheader("Explainable chronic disease risk prediction")

    if is_logged_in():
        user = get_current_user()
        st.success(f"Welcome back, **{user.username}**! Use the sidebar to navigate.")
    else:
        st.info("👈 Log in or create an account from the sidebar to get started.")

    st.markdown(
        """
        Welcome to **HealthGuardian AI** — a portfolio project demonstrating
        end-to-end ML engineering: authentication, disease risk prediction,
        SHAP-based explainability, and PDF reporting.
        """
    )

    with st.expander("Roadmap", expanded=True):
        st.markdown(
            """
            - [x] Milestone 1: Project skeleton, dependencies, first commit
            - [x] Milestone 2: Authentication
            - [ ] Milestone 3: SQLite schema (prediction history)
            - [ ] Milestone 4: Heart disease model
            - [ ] Milestone 5: Prediction UI integration
            - [ ] Milestone 6: SHAP explanations
            - [ ] Milestone 7: Diabetes model
            - [ ] Milestone 8: Dashboards and PDF reports
            - [ ] Milestone 9: Testing and deployment
            """
        )


if __name__ == "__main__":
    render_landing()
