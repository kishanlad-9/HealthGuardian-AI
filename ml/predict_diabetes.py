"""Inference logic for the diabetes model - same pattern as ml/predict_heart.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

import joblib
import pandas as pd

from ml.preprocessing_diabetes import ALL_FEATURES

MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "diabetes_model.joblib"

# Same tertile thresholds as heart disease (ml/predict_heart.py) for
# consistency across the app - not derived from any clinical guideline.
LOW_RISK_THRESHOLD = 0.33
HIGH_RISK_THRESHOLD = 0.66


class ModelNotTrainedError(Exception):
    pass


@dataclass(frozen=True)
class DiabetesPredictionResult:
    risk_probability: float
    risk_label: str
    model_version: str


def _load_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"{MODEL_PATH} not found. Run: python ml/download_diabetes_dataset.py "
            "&& python ml/train_diabetes_model.py"
        )
    return joblib.load(MODEL_PATH)


def probability_to_risk_label(probability: float) -> str:
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    if probability < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def predict_diabetes(input_features: dict) -> DiabetesPredictionResult:
    """input_features must contain exactly the keys in ALL_FEATURES:
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
    diabetes_pedigree, age. Pass raw measurements including any 0 values
    as-is - the pipeline's imputer handles the zero-as-missing convention
    for glucose/blood_pressure/skin_thickness/insulin/bmi internally."""
    bundle = _load_model()
    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]

    missing = set(ALL_FEATURES) - set(input_features.keys())
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    X = pd.DataFrame([{f: input_features[f] for f in ALL_FEATURES}])
    probability = float(pipeline.predict_proba(X)[0, 1])

    return DiabetesPredictionResult(
        risk_probability=round(probability, 4),
        risk_label=probability_to_risk_label(probability),
        model_version=f"diabetes_{model_name}_v1",
    )
