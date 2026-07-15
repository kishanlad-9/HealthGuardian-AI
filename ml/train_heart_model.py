"""
Trains and compares three classifiers for heart disease risk prediction,
selects the best by cross-validated ROC-AUC, evaluates it on a held-out
test set, and serializes the winning pipeline (preprocessing + model as
one artifact - see preprocessing.build_pipeline) to saved_models/.

Why compare three models instead of just picking XGBoost: on a small
dataset (303 rows), a simpler model can legitimately outperform a more
complex one, and it's worth actually checking rather than assuming the
fanciest tool wins. Logistic Regression is included as an interpretable
baseline that a portfolio project should be able to justify losing to,
not just skip.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from ml.preprocessing import ALL_FEATURES, build_pipeline, load_heart_data

MODEL_DIR = Path(__file__).parent.parent / "saved_models"
REPORT_PATH = Path(__file__).parent.parent / "reports" / "heart_model_training_report.json"

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
}


def compare_models(X_train, y_train) -> dict[str, float]:
    """5-fold stratified cross-validation on the training set only (test
    set stays untouched until final evaluation, to avoid leakage into
    model selection). Returns mean ROC-AUC per candidate."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, classifier in CANDIDATES.items():
        pipeline = build_pipeline(classifier)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        scores[name] = {"mean_roc_auc": round(cv_scores.mean(), 4), "std_roc_auc": round(cv_scores.std(), 4)}
        print(f"{name:<22} CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return scores


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def main():
    X, y = load_heart_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows (stratified 80/20 split)\n")

    print("=== Cross-validated model comparison (train set only) ===")
    cv_scores = compare_models(X_train, y_train)

    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean_roc_auc"])
    print(f"\nBest by CV ROC-AUC: {best_name}")

    print(f"\n=== Refitting {best_name} on full train set, evaluating on held-out test set ===")
    best_pipeline = build_pipeline(CANDIDATES[best_name])
    best_pipeline.fit(X_train, y_train)
    test_metrics = evaluate_on_test(best_pipeline, X_test, y_test)

    print(f"Test accuracy:  {test_metrics['accuracy']}")
    print(f"Test precision: {test_metrics['precision']}")
    print(f"Test recall:    {test_metrics['recall']}")
    print(f"Test F1:        {test_metrics['f1_score']}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(
        {"pipeline": best_pipeline, "feature_names": ALL_FEATURES, "model_name": best_name},
        model_path,
    )
    print(f"\nSaved model -> {model_path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model_selected": best_name,
        "cross_validation_scores": cv_scores,
        "test_set_metrics": test_metrics,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved training report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
