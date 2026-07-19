# saved_models/

Serialized, trained model artifacts (`.joblib`), loaded at runtime by the
Streamlit app. Not committed to git — regenerate with the training scripts
below rather than committing binaries to the repo.

## heart_disease_model.joblib (Milestone 4, corrected in Milestone 5)

```bash
python ml/download_heart_dataset.py
python ml/train_heart_model.py
```

**⚠️ History**: the model trained in Milestone 4 was trained on inverted
labels (this dataset's raw `target` column has disease-present = `0`, the
opposite of the intuitive convention). Caught in Milestone 5 by
sanity-checking predictions against two synthetic patient profiles - a
healthy profile scored *higher* risk than an unhealthy one, which was the
tell. Fixed in `ml/preprocessing.load_heart_data()`; a regression test
now guards against this recurring.

**Current result**: Random Forest won cross-validation (0.910 ROC-AUC)
over Logistic Regression (0.906) and XGBoost (0.887). Test set: 82.0%
accuracy, 0.903 ROC-AUC, 75.0% recall.

## diabetes_model.joblib (Milestone 7)

```bash
python ml/download_diabetes_dataset.py
python ml/train_diabetes_model.py
```

Same methodology as the heart disease model: compares Logistic Regression,
Random Forest, and XGBoost via 5-fold cross-validated ROC-AUC, refits the
winner, evaluates on a held-out 20% test set. Uses `SimpleImputer` for the
zero-as-missing data quality issue (see `datasets/README.md`) instead of
one-hot encoding (diabetes has no categorical features, unlike heart disease).

**Result**: **Logistic Regression** won cross-validation (0.843 ROC-AUC)
over Random Forest (0.820) and XGBoost (0.779). Test set: 70.8% accuracy,
0.813 ROC-AUC, 50.0% recall. Verified class direction is correct (not
inverted like the heart dataset) against two synthetic profiles: a
healthy profile scored 1.9%, a multi-risk-factor profile scored 97.5%.

Note the lower recall (50%) compared to heart disease (75%) - worth being
upfront about: this model misses roughly half of actual positive cases on
the held-out test set. Acceptable for a portfolio project demonstrating
the ML pipeline, but a real deployment would need to either accept a
higher false-positive rate (lower the decision threshold) or gather more
training data before being trusted for anything consequential.

Because Logistic Regression (not a tree model) won here, `ml/explain_diabetes.py`
uses a different, model-agnostic SHAP explainer than the heart disease
module's `TreeExplainer` — see that file's docstring for why.

Both saved files are dicts: `{"pipeline": ..., "feature_names": [...],
"model_name": "..."}` — load with `joblib.load()`, or use the corresponding
`ml/predict_*.py` module's ready-made inference function.
