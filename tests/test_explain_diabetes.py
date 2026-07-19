"""Unit tests for ml/explain_diabetes.py."""
import pytest

from ml.explain_diabetes import MODEL_PATH, ExplanationResult, explain_prediction
from ml.predict_diabetes import predict_diabetes

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained - run: python ml/train_diabetes_model.py",
)

PROFILE = {
    "pregnancies": 6, "glucose": 190, "blood_pressure": 90, "skin_thickness": 40,
    "insulin": 300, "bmi": 40.0, "diabetes_pedigree": 1.5, "age": 55,
}


def test_explain_returns_all_8_features():
    result = explain_prediction(PROFILE)
    assert isinstance(result, ExplanationResult)
    assert len(result.contributions) == 8
    assert {c.feature for c in result.contributions} == set(PROFILE.keys())


def test_contributions_sorted_by_magnitude_descending():
    result = explain_prediction(PROFILE)
    magnitudes = [abs(c.shap_value) for c in result.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_values_are_additive_to_actual_prediction():
    """Same core correctness check as test_explain_heart.py - base_value +
    all contributions must reconstruct the actual predicted probability."""
    result = explain_prediction(PROFILE)
    prediction = predict_diabetes(PROFILE)
    reconstructed = result.base_value + sum(c.shap_value for c in result.contributions)
    assert reconstructed == pytest.approx(prediction.risk_probability, abs=0.02)


def test_glucose_is_top_contributor_for_high_glucose_profile():
    """Domain sanity check: glucose is the primary diagnostic criterion
    for diabetes - for a profile with clearly elevated glucose (190, vs.
    a healthy ~85-100), it should be the single strongest contributor."""
    result = explain_prediction(PROFILE)
    assert result.top(1)[0].feature == "glucose"
    assert result.top(1)[0].shap_value > 0
