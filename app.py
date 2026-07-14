"""
HealthGuardian AI - main Streamlit entry point.

This file stays intentionally thin: it configures the page and renders a
landing view. Actual feature pages live under pages/ (Streamlit's
multi-page-app convention - any .py file there becomes a sidebar page
automatically). Business logic belongs in authentication/, ml/, database/,
and utils/ - not here - so this file remains easy to read as the app grows.
"""
import streamlit as st

st.set_page_config(
    page_title="HealthGuardian AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_landing() -> None:
    """Render the landing page shown before any feature milestones exist."""
    st.title("🩺 HealthGuardian AI")
    st.subheader("Explainable chronic disease risk prediction")

    st.markdown(
        """
        Welcome to **HealthGuardian AI** — a portfolio project demonstrating
        end-to-end ML engineering: authentication, disease risk prediction,
        SHAP-based explainability, and PDF reporting.

        **Project status: Milestone 1 - project skeleton.**
        Feature pages (login, prediction, dashboard) will appear in the
        sidebar as later milestones are completed.
        """
    )

    with st.expander("Roadmap", expanded=True):
        st.markdown(
            """
            - [x] Milestone 1: Project skeleton, dependencies, first commit
            - [ ] Milestone 2: Authentication
            - [ ] Milestone 3: SQLite schema
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
