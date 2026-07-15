# HealthGuardian AI

Explainable chronic disease risk prediction — a portfolio project
demonstrating end-to-end ML engineering: authentication, ML model training
and serving, SHAP-based explainability, and PDF reporting, built as a
Streamlit application.

**Project status:** Milestone 4 of 9 — heart disease model trained. See [Roadmap](#roadmap).

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
├── app.py                    # Streamlit entry point (thin - see docstring)
├── requirements.txt           # runtime dependencies
├── requirements-dev.txt       # + pytest, for running tests
├── .streamlit/config.toml     # theme + server config
├── database/                  # SQLite schema, connection, queries
├── authentication/            # signup/login, password hashing, session state
├── pages/                     # Streamlit multi-page-app pages (auto-detected)
├── ml/                        # preprocessing, training scripts, model classes
├── datasets/                  # raw/processed data (gitignored, see datasets/README.md)
├── saved_models/               # trained model artifacts (gitignored, see saved_models/README.md)
├── reports/                    # training reports + generated PDF reports (gitignored)
├── utils/                      # shared helpers (validation, formatting, etc.)
└── tests/                      # unit tests (pytest)
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
- [x] **Milestone 2** — Authentication (signup/login, bcrypt, session state)
- [x] **Milestone 3** — SQLite schema (users, prediction history)
- [x] **Milestone 4** — Heart disease model (UCI dataset, training pipeline)
- [ ] **Milestone 5** — Prediction UI integration
- [ ] **Milestone 6** — SHAP explanations
- [ ] **Milestone 7** — Diabetes model (Pima Indians dataset)
- [ ] **Milestone 8** — Dashboards and PDF reports
- [ ] **Milestone 9** — Testing and deployment

## Milestone 4: Heart Disease Model

```bash
python ml/download_heart_dataset.py    # downloads datasets/heart_disease_uci.csv
python -m ml.train_heart_model         # trains, compares, saves the best model
```

Compares Logistic Regression, Random Forest, and XGBoost via 5-fold
cross-validated ROC-AUC on the UCI Cleveland Heart Disease dataset (303
patients), then evaluates the winner on a held-out 20% test set.

**Result: Logistic Regression won** (0.892 CV ROC-AUC) over Random Forest
(0.887) and XGBoost (0.869) — a useful reminder that the more complex model
doesn't automatically win, especially at this dataset size. Test set: 88.5%
accuracy, 0.910 ROC-AUC, 0.939 recall. Full metrics in
`reports/heart_model_training_report.json`; narrative writeup in
`saved_models/README.md`.

27/27 tests passing: `pytest tests/` (requires `requirements-dev.txt`).

## Git Strategy

`main` (stable) ← `development` ← `feature/*` branches, one per milestone
area (`feature/authentication`, `feature/ml-heart`, etc.). Commits are
granular and scoped to one logical change each.

## License

TBD.
