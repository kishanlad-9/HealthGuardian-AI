"""Unit tests for ml/explain_heart.py. Requires the model to be trained -
skipped with a clear reason if not (same pattern as test_predict_heart.py)."""
import pytest

from ml.explain_heart import MODEL_PATH, ExplanationResult, explain_prediction
from ml.predict_heart import predict_heart_disease

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained - run: python ml/train_heart_model.py",
)

LOW_RISK_PROFILE = {
    "age": 35, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
    "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0,
    "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2,
}


def test_explain_returns_all_13_features():
    result = explain_prediction(LOW_RISK_PROFILE)
    assert isinstance(result, ExplanationResult)
    assert len(result.contributions) == 13
    assert {c.feature for c in result.contributions} == set(LOW_RISK_PROFILE.keys())


def test_contributions_sorted_by_magnitude_descending():
    result = explain_prediction(LOW_RISK_PROFILE)
    magnitudes = [abs(c.shap_value) for c in result.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_values_are_additive_to_actual_prediction():
    """The core correctness check for this module: base_value + sum of all
    per-feature SHAP contributions must reconstruct the model's actual
    predicted probability. If the one-hot aggregation logic in
    explain_prediction() has a bug (wrong column mapping, double-counting,
    a dropped column), this is what would catch it - the numbers would
    stop adding up even if each individual value looked plausible."""
    result = explain_prediction(LOW_RISK_PROFILE)
    prediction = predict_heart_disease(LOW_RISK_PROFILE)

    reconstructed = result.base_value + sum(c.shap_value for c in result.contributions)
    assert reconstructed == pytest.approx(prediction.risk_probability, abs=0.01)


def test_top_n_returns_requested_count():
    result = explain_prediction(LOW_RISK_PROFILE)
    assert len(result.top(3)) == 3
    assert len(result.top(20)) == 13  # can't return more than exist


def test_direction_property_matches_sign():
    result = explain_prediction(LOW_RISK_PROFILE)
    for c in result.contributions:
        if c.shap_value > 0:
            assert c.direction == "increased"
        elif c.shap_value < 0:
            assert c.direction == "decreased"
