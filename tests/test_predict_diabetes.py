"""Unit tests for ml/predict_diabetes.py. Requires the model to be trained."""
import pytest

from ml.predict_diabetes import (
    DiabetesPredictionResult,
    MODEL_PATH,
    predict_diabetes,
    probability_to_risk_label,
)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained - run: python ml/train_diabetes_model.py",
)

VALID_INPUT = {
    "pregnancies": 2, "glucose": 120, "blood_pressure": 70, "skin_thickness": 25,
    "insulin": 100, "bmi": 28.0, "diabetes_pedigree": 0.4, "age": 35,
}


def test_predict_returns_valid_result():
    result = predict_diabetes(VALID_INPUT)
    assert isinstance(result, DiabetesPredictionResult)
    assert 0.0 <= result.risk_probability <= 1.0
    assert result.risk_label in {"Low", "Medium", "High"}
    assert result.model_version.startswith("diabetes_")


def test_predict_raises_on_missing_feature():
    incomplete = {k: v for k, v in VALID_INPUT.items() if k != "glucose"}
    with pytest.raises(KeyError, match="glucose"):
        predict_diabetes(incomplete)


def test_probability_to_risk_label_boundaries():
    assert probability_to_risk_label(0.0) == "Low"
    assert probability_to_risk_label(0.33) == "Medium"
    assert probability_to_risk_label(0.66) == "High"


def test_low_risk_profile_scores_lower_than_high_risk_profile():
    """Regression guard, same pattern as the heart disease test: asserts
    correct relative ordering, not an exact label - the property that
    actually matters."""
    low_risk_profile = {
        "pregnancies": 1, "glucose": 85, "blood_pressure": 66, "skin_thickness": 29,
        "insulin": 80, "bmi": 22.0, "diabetes_pedigree": 0.2, "age": 25,
    }
    high_risk_profile = {
        "pregnancies": 6, "glucose": 190, "blood_pressure": 90, "skin_thickness": 40,
        "insulin": 300, "bmi": 40.0, "diabetes_pedigree": 1.5, "age": 55,
    }
    low_result = predict_diabetes(low_risk_profile)
    high_result = predict_diabetes(high_risk_profile)
    assert high_result.risk_probability > low_result.risk_probability
