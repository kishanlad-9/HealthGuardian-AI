# saved_models/

Serialized, trained model artifacts (`.joblib`), loaded at runtime by the
Streamlit app. Not committed to git — regenerate with the training scripts
below rather than committing binaries to the repo.

## heart_disease_model.joblib (Milestone 4, corrected in Milestone 5)

```bash
python ml/download_heart_dataset.py   # if not already downloaded
python -m ml.train_heart_model
```

Trains and compares Logistic Regression, Random Forest, and XGBoost via
5-fold cross-validated ROC-AUC on the training split, refits the winner on
the full training set, evaluates on a held-out 20% test set, and saves the
winning **pipeline** (preprocessing + model as one artifact — see
`ml/preprocessing.py`) here.

**⚠️ History**: the model trained in Milestone 4 was trained on inverted
labels (this dataset's raw `target` column has disease-present = `0`, the
opposite of the intuitive convention — see `datasets/README.md`). Caught in
Milestone 5 by sanity-checking predictions against two synthetic patient
profiles before wiring the model into the UI — a healthy profile scored
*higher* risk than an unhealthy one, which was the tell. Fixed in
`ml/preprocessing.load_heart_data()` and retrained; a regression test now
guards against this recurring
(`tests/test_preprocessing.py::test_target_direction_matches_known_risk_factors`).

**Current result** (post-fix, see `reports/heart_model_training_report.json`
for full metrics): **Random Forest** won cross-validation (0.910 ROC-AUC)
over Logistic Regression (0.906) and XGBoost (0.887). Test set: 82.0%
accuracy, 0.903 ROC-AUC, 75.0% recall.

The saved file is a dict: `{"pipeline": ..., "feature_names": [...],
"model_name": "random_forest"}` — load with `joblib.load()`, or use
`ml/predict_heart.py`'s `predict_heart_disease()` for a ready-made
inference function that also applies risk-level labeling.
