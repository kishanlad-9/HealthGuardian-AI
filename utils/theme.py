"""
Centralized theme for the whole app - one CSS injection function, called
at the top of every page, instead of copy-pasting <style> blocks. Keeps
the palette defined in exactly one place (the CSS custom properties at
the top of THEME_CSS) so a future color change is a one-file edit.

Palette rationale (blended from two reference styles the user provided):
  - Deep navy/indigo primary (--brand-navy) - the dominant color in both
    references, reads as trustworthy/clinical without being cold.
  - Indigo accent (--brand-indigo) for primary actions/buttons - matches
    the reference login page's button color, distinct enough from the
    navy to show interactivity.
  - Mint (--brand-mint) as a small-dose accent for tags/success states -
    lifted from the first reference's "Explore Programs" pill and stat
    highlights. Used sparingly, not as a second primary color.
  - Soft lavender-gray page background (--bg-page) instead of plain
    white, matching the second reference's card-on-tinted-background look.
"""
import streamlit as st

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --brand-navy: #1B2A56;
    --brand-navy-dark: #131F42;
    --brand-indigo: #5B5FEF;
    --brand-indigo-hover: #4A4EDC;
    --brand-mint: #4FD1AE;
    --brand-mint-light: #E3FBF3;
    --bg-page: #F6F7FC;
    --bg-card: #FFFFFF;
    --text-heading: #16204A;
    --text-body: #5B6178;
    --border-soft: #E4E7F2;
    --shadow-soft: 0 4px 20px rgba(27, 42, 86, 0.08);
    --shadow-lift: 0 10px 30px rgba(27, 42, 86, 0.16);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-heading) !important;
    font-weight: 600 !important;
}

/* Page background */
[data-testid="stAppViewContainer"] {
    background: var(--bg-page);
}
[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--brand-navy);
}
[data-testid="stSidebar"] * {
    color: #E8EAF7 !important;
}
[data-testid="stSidebar"] a:hover {
    color: var(--brand-mint) !important;
}

/* Buttons - lift + shadow on hover, smooth transition */
.stButton > button, .stFormSubmitButton > button {
    background: var(--brand-indigo) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease !important;
    box-shadow: var(--shadow-soft) !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: var(--brand-indigo-hover) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lift) !important;
}
.stButton > button:active, .stFormSubmitButton > button:active {
    transform: translateY(0) !important;
}

/* Text inputs, number inputs, selectboxes - soft rounded, glow on focus */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid var(--border-soft) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--brand-indigo) !important;
    box-shadow: 0 0 0 3px rgba(91, 95, 239, 0.15) !important;
}

/* Cards - used by custom containers (login card, metric wrappers, etc.) */
.hg-card {
    background: var(--bg-card);
    border-radius: 20px;
    box-shadow: var(--shadow-soft);
    padding: 2.5rem;
    transition: box-shadow 0.2s ease;
    animation: hg-fade-in-up 0.5s ease both;
}
.hg-card:hover {
    box-shadow: var(--shadow-lift);
}

/* Tags/pills (mint accent, sparingly - matches reference "Explore Programs" pill) */
.hg-pill {
    display: inline-block;
    background: var(--brand-mint-light);
    color: #0E9F79;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}

/* Metric widget polish */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: var(--shadow-soft);
    transition: transform 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
}

/* Progress bar - brand color instead of default red */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--brand-indigo), var(--brand-mint)) !important;
}

/* Fade/slide-in animation for main content on load */
[data-testid="stAppViewContainer"] .main .block-container {
    animation: hg-fade-in-up 0.4s ease both;
}
@keyframes hg-fade-in-up {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes hg-float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
@keyframes hg-pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.06); opacity: 0.85; }
}

/* Forms rendered as cards (login/signup, prediction forms) */
[data-testid="stForm"] {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 2rem 2rem 1.2rem 2rem;
    box-shadow: var(--shadow-soft);
    border: 1px solid var(--border-soft);
    animation: hg-fade-in-up 0.5s ease both;
}

/* Tabs (used on the login page) */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: var(--text-body);
}
.stTabs [aria-selected="true"] {
    color: var(--brand-indigo) !important;
}
</style>
"""


def inject_theme() -> None:
    """Call once near the top of every page (after st.set_page_config)."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
