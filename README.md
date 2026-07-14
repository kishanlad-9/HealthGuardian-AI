# HealthGuardian AI

Explainable chronic disease risk prediction — a portfolio project
demonstrating end-to-end ML engineering: authentication, ML model training
and serving, SHAP-based explainability, and PDF reporting, built as a
Streamlit application.

**Project status:** Milestone 1 of 9 — project skeleton. See [Roadmap](#roadmap).

## Tech Stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Plotly |
| Persistence | SQLite |
| Auth | bcrypt (password hashing) |
| Reporting | ReportLab (PDF) |

## Project Structure

```
HealthGuardian-AI/
├── app.py                 # Streamlit entry point (thin - see docstring)
├── requirements.txt
├── .streamlit/config.toml # theme + server config
├── database/               # SQLite schema, connection, queries
├── authentication/         # signup/login, password hashing, session state
├── pages/                  # Streamlit multi-page-app pages (auto-detected)
├── ml/                     # training scripts, preprocessing, model classes
├── datasets/                # raw/processed data (gitignored, see datasets/README.md)
├── saved_models/            # trained model artifacts (gitignored)
├── reports/                 # generated PDF reports (gitignored)
├── utils/                   # shared helpers (validation, formatting, etc.)
└── tests/                   # unit tests
```

## Setup

```bash
git clone <repo-url>
cd HealthGuardian-AI
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Visit `http://localhost:8501`.

## Roadmap

- [x] **Milestone 1** — Project skeleton, dependencies, first commit
- [ ] **Milestone 2** — Authentication (signup/login, bcrypt, session state)
- [ ] **Milestone 3** — SQLite schema (users, prediction history)
- [ ] **Milestone 4** — Heart disease model (UCI dataset, training pipeline)
- [ ] **Milestone 5** — Prediction UI integration
- [ ] **Milestone 6** — SHAP explanations
- [ ] **Milestone 7** — Diabetes model (Pima Indians dataset)
- [ ] **Milestone 8** — Dashboards and PDF reports
- [ ] **Milestone 9** — Testing and deployment

## Git Strategy

`main` (stable) ← `development` ← `feature/*` branches, one per milestone
area (`feature/authentication`, `feature/ml-heart`, etc.). Commits are
granular and scoped to one logical change each.

## License

TBD.
