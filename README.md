# HealthGuardian AI

Explainable chronic disease risk prediction — a portfolio project
demonstrating end-to-end ML engineering: authentication, ML model training
and serving, SHAP-based explainability, and PDF reporting, built as a
Streamlit application.

**Project status:** Milestone 5 of 9 — prediction UI live. See [Roadmap](#roadmap).

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
- [x] **Milestone 5** — Prediction UI integration
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
patients), then evaluates the winner on a held-out 20% test set. See
Milestone 5 below for a critical label-direction bug found and fixed after
this milestone — numbers here reflect the corrected model.

**Result: Random Forest won** (0.910 CV ROC-AUC) over Logistic Regression
(0.906) and XGBoost (0.887). Full metrics in
`reports/heart_model_training_report.json`; narrative writeup in
`saved_models/README.md`.

## Milestone 5: Prediction UI Integration

`pages/2_Heart_Disease_Prediction.py` — a login-gated form covering all 13
clinical features, calls the trained model, shows the risk level and
probability, saves the result to `prediction_history`, and lists the
user's recent predictions below the form.

**Critical bug found and fixed while wiring this up**: sanity-checking the
model against two synthetic profiles (a healthy 35-year-old vs. a 65-year-old
with multiple risk factors) produced *backwards* results — the healthy
profile scored high risk. Investigation confirmed the dataset's raw `target`
column is inverted in this specific file (`0` = disease present, `1` =
disease absent — a known, documented quirk of this dataset's lineage, see
`datasets/README.md`), verified independently against three separate
clinical indicators (age, max heart rate, exercise-induced angina rate) in
the raw data before trusting it. Fixed in `ml/preprocessing.load_heart_data()`
(labels now inverted so `1` consistently means "disease present" everywhere
downstream) and model retrained. A regression test
(`test_target_direction_matches_known_risk_factors`) now guards against this
silently coming back. **Model changed as a result: Random Forest now wins**
cross-validation (0.910 ROC-AUC) over Logistic Regression (0.906) and XGBoost
(0.887) — the earlier Milestone 4 result was measured on inverted labels and
should be considered superseded.

31/31 tests passing.

## Git Strategy

`main` (stable) ← `development` ← `feature/*` branches, one per milestone
area (`feature/authentication`, `feature/ml-heart`, etc.). Commits are
granular and scoped to one logical change each.

## License

TBD.
