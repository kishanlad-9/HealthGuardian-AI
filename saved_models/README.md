# saved_models/

Serialized, trained model artifacts (`.joblib`), loaded at runtime by the
Streamlit app. Not committed to git — regenerate with the training scripts
below rather than committing binaries to the repo.

## heart_disease_model.joblib (Milestone 4)

```bash
python ml/download_heart_dataset.py   # if not already downloaded
python -m ml.train_heart_model
```

Trains and compares Logistic Regression, Random Forest, and XGBoost via
5-fold cross-validated ROC-AUC on the training split, refits the winner on
the full training set, evaluates on a held-out 20% test set, and saves the
winning **pipeline** (preprocessing + model as one artifact — see
`ml/preprocessing.py`) here.

Current result (see `reports/heart_model_training_report.json` for full
metrics): **Logistic Regression** won cross-validation (0.892 ROC-AUC) over
Random Forest (0.887) and XGBoost (0.869) — worth noting since it's a
reminder that the more complex model doesn't automatically win, especially
on a 303-row dataset. Test set: 88.5% accuracy, 0.910 ROC-AUC.

The saved file is a dict: `{"pipeline": ..., "feature_names": [...],
"model_name": "logistic_regression"}` — load with `joblib.load()`.
