# HealthGuardian AI

Explainable chronic disease risk prediction — a portfolio project
demonstrating end-to-end ML engineering: authentication, ML model training
and serving, SHAP-based explainability, and PDF reporting, built as a
Streamlit application.

**Project status:** Milestone 7 of 9 — diabetes model live. See [Roadmap](#roadmap).

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
- [x] **Milestone 6** — SHAP explanations
- [x] **Milestone 7** — Diabetes model (Pima Indians dataset)
- [ ] **Milestone 8** — Dashboards and PDF reports
- [ ] **Milestone 9** — Testing and deployment

## Milestone 7: Diabetes Model

```bash
python ml/download_diabetes_dataset.py    # downloads datasets/pima_diabetes.csv
python ml/train_diabetes_model.py         # trains, compares, saves the best model
```

Built the full diabetes vertical in one pass (data → training → prediction →
SHAP explanation → UI), reusing the pattern established for heart disease:
`pages/3_Diabetes_Prediction.py` — login-gated form, prediction, plain-language
+ chart explanation, saved to prediction history. No schema changes needed;
`prediction_history.disease_type` already allowed `'diabetes'` since Milestone 3.

**A real data-quality issue, handled not just noted**: this dataset uses `0`
as a missing-value placeholder in `glucose`, `blood_pressure`,
`skin_thickness`, `insulin`, and `bmi` — verified empirically (insulin missing
in 48.7% of rows). Fixed with median imputation inside the training pipeline
(fit on the training fold only). `pregnancies=0` is a legitimate value and
was deliberately left untouched.

**Result: Logistic Regression won** (0.843 CV ROC-AUC) over Random Forest
(0.820) and XGBoost (0.779). Test set: 70.8% accuracy, 0.813 ROC-AUC, 50.0%
recall — lower than the heart model's recall, documented honestly in
`saved_models/README.md` rather than glossed over.

Because the winning model is linear (not a tree ensemble like heart
disease's Random Forest), `ml/explain_diabetes.py` uses a different,
model-agnostic SHAP explainer instead of `TreeExplainer` - see that file's
docstring. Verified class-direction correctness this time before trusting
it (unlike heart disease, no inversion here) and confirmed SHAP additivity
holds.

51/51 tests passing.

## Milestone 6: SHAP Explanations

Every prediction now comes with a plain-English explanation of *why* -
not just a risk score. `ml/explain_heart.py` uses `shap.TreeExplainer`
(exact SHAP values for the tree-based Random Forest model, not the slower
model-agnostic approximation) to compute each feature's contribution,
aggregates the one-hot-encoded columns back to the original 13 features
the user actually filled in (so you see "Chest pain type" once, not four
separate `cp_0`/`cp_1`/`cp_2`/`cp_3` dummy columns), and surfaces the top
contributors as both plain-language sentences and a Plotly bar chart on
the prediction page.

**Correctness check that matters here**: SHAP values are only meaningful
if they're *additive* - base value + all contributions must reconstruct
the actual predicted probability exactly. Verified this holds
(`tests/test_explain_heart.py::test_shap_values_are_additive_to_actual_prediction`)
before trusting any of the individual numbers.

**A counterintuitive-but-verified finding, not a bug**: "Asymptomatic"
chest pain *decreases* predicted risk in this model. Checked directly
against the training data before accepting it: in this specific dataset,
patients with `cp=0` ("typical angina") have a 72.7% disease rate vs. only
30.4% for `cp=3` ("asymptomatic") — a well-documented quirk of this exact
Cleveland dataset, not a modeling error. Documented in
`ml/explain_heart.py` so it doesn't get "fixed" incorrectly later.

`prediction_history.shap_values` (already in the Milestone 3 schema,
unused until now) stores the full explanation alongside each prediction.

37/37 tests passing.

## Milestone 4: Heart Disease Model

```bash
python ml/download_heart_dataset.py    # downloads datasets/heart_disease_uci.csv
python -m ml.train_heart_model         # trains, compares, saves the best model
```

Both `python ml/train_heart_model.py` (direct) and `python -m ml.train_heart_model`
(module) work — the script inserts the project root onto `sys.path` itself,
so VS Code's "Run" button works too without a `ModuleNotFoundError`.

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
