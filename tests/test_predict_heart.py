"""Unit tests for ml/predict_heart.py. Requires the model to already be
trained (python -m ml.train_heart_model) - skipped with a clear reason if not."""
import pytest

from ml.predict_heart import (
    HeartPredictionResult,
    MODEL_PATH,
    ModelNotTrainedError,
    predict_heart_disease,
    probability_to_risk_label,
)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained - run: python -m ml.train_heart_model",
)

VALID_INPUT = {
    "age": 50, "sex": 1, "cp": 0, "trestbps": 130, "chol": 230,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
}


def test_predict_returns_valid_result():
    result = predict_heart_disease(VALID_INPUT)
    assert isinstance(result, HeartPredictionResult)
    assert 0.0 <= result.risk_probability <= 1.0
    assert result.risk_label in {"Low", "Medium", "High"}
    assert result.model_version.startswith("heart_")


def test_predict_raises_on_missing_feature():
    incomplete = {k: v for k, v in VALID_INPUT.items() if k != "age"}
    with pytest.raises(KeyError, match="age"):
        predict_heart_disease(incomplete)


def test_probability_to_risk_label_boundaries():
    assert probability_to_risk_label(0.0) == "Low"
    assert probability_to_risk_label(0.32) == "Low"
    assert probability_to_risk_label(0.33) == "Medium"
    assert probability_to_risk_label(0.65) == "Medium"
    assert probability_to_risk_label(0.66) == "High"
    assert probability_to_risk_label(1.0) == "High"


def test_low_risk_profile_scores_lower_than_high_risk_profile():
    """Regression guard for the target-inversion bug: a profile with
    multiple classic risk factors should score meaningfully higher than
    a profile with none, in the correct direction. Doesn't assert exact
    labels (model behavior can shift slightly on retraining) - asserts
    the *relative ordering*, which is the property that actually matters
    and is what the earlier bug got backwards."""
    low_risk_profile = {
        "age": 35, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0,
        "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2,
    }
    high_risk_profile = {
        "age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 300,
        "fbs": 1, "restecg": 1, "thalach": 110, "exang": 1,
        "oldpeak": 3.0, "slope": 0, "ca": 3, "thal": 3,
    }
    low_result = predict_heart_disease(low_risk_profile)
    high_result = predict_heart_disease(high_risk_profile)
    assert high_result.risk_probability > low_result.risk_probability
