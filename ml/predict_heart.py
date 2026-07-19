"""
Inference logic for the heart disease model - the layer between the
saved .joblib pipeline and the Streamlit UI.

Kept separate from pages/ (Streamlit) for the same reason authentication/auth.py
is separate from session.py: this module has zero Streamlit dependency, so
it's directly unit-testable without mocking Streamlit, and could be reused
by a future non-Streamlit interface (e.g. an API) without changes.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

# See train_heart_model.py for why this is needed - makes direct execution
# and VS Code's "Run" button work the same as `python -m ml.predict_heart`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from ml.preprocessing import ALL_FEATURES

MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "heart_disease_model.joblib"

# Risk-label thresholds on predicted probability of disease presence.
# Chosen as simple, even tertiles rather than derived from any clinical
# guideline - this is a portfolio project, not a diagnostic tool, and the
# UI says so (see pages/2_Heart_Disease_Prediction.py). A real clinical
# deployment would need these calibrated against actual outcome data.
LOW_RISK_THRESHOLD = 0.33
HIGH_RISK_THRESHOLD = 0.66


class ModelNotTrainedError(Exception):
    """Raised when saved_models/heart_disease_model.joblib doesn't exist yet."""


@dataclass(frozen=True)
class HeartPredictionResult:
    risk_probability: float
    risk_label: str
    model_version: str


def _load_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"{MODEL_PATH} not found. Run: python ml/download_heart_dataset.py "
            "&& python -m ml.train_heart_model"
        )
    return joblib.load(MODEL_PATH)


def probability_to_risk_label(probability: float) -> str:
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    if probability < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def predict_heart_disease(input_features: dict) -> HeartPredictionResult:
    """input_features must contain exactly the keys in ALL_FEATURES
    (see ml/preprocessing.py) - age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak, slope, ca, thal.

    Raises ModelNotTrainedError if the model hasn't been trained yet, and
    KeyError (via the DataFrame construction) if a required feature is
    missing - both are meant to fail loudly rather than silently predict
    on incomplete/wrong input.
    """
    bundle = _load_model()
    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]

    missing = set(ALL_FEATURES) - set(input_features.keys())
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    X = pd.DataFrame([{feature: input_features[feature] for feature in ALL_FEATURES}])
    probability = float(pipeline.predict_proba(X)[0, 1])

    return HeartPredictionResult(
        risk_probability=round(probability, 4),
        risk_label=probability_to_risk_label(probability),
        model_version=f"heart_{model_name}_v1",
    )
