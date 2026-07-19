"""
SHAP-based explanations for individual diabetes predictions.

Design decision, different from ml/explain_heart.py: the diabetes model
picked by ml/train_diabetes_model.py is Logistic Regression (see
saved_models/README.md), not a tree ensemble, so shap.TreeExplainer isn't
applicable. Rather than special-case each model type, this module treats
the whole pipeline's predict_proba as a black-box function and uses
shap.Explainer's model-agnostic permutation-based backend. Two consequences,
both deliberate:
  1. Contributions come out already in PROBABILITY space (matching how
     heart disease explanations are presented - additivity holds against
     the actual predicted probability, not log-odds), rather than needing
     a separate unit system for this disease.
  2. This module keeps working unchanged even if a future retrain picks a
     different model type - the heart-disease module is coupled to
     "the model is a tree ensemble"; this one isn't coupled to any
     specific model family, at the cost of being slower per explanation
     (permutation sampling vs. TreeExplainer's exact polynomial-time
     computation) - acceptable for a single on-demand prediction, not
     something to use in a hot loop.

No one-hot aggregation needed here (unlike heart disease): all 8 diabetes
features are numeric, so SHAP's feature space already matches the
features the user actually filled in - a 1:1 mapping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import shap

from ml.feature_labels import DIABETES_FEATURE_DISPLAY_NAMES
from ml.preprocessing_diabetes import ALL_FEATURES, load_diabetes_data

MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "diabetes_model.joblib"
BACKGROUND_SAMPLE_SIZE = 100  # rows sampled from the training data as the SHAP background/masker


class ModelNotTrainedError(Exception):
    pass


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    display_name: str
    value_display: str
    shap_value: float  # positive = pushed prediction toward "diabetes present"

    @property
    def direction(self) -> str:
        return "increased" if self.shap_value > 0 else "decreased"


@dataclass(frozen=True)
class ExplanationResult:
    base_value: float
    contributions: list[FeatureContribution]

    def top(self, n: int = 5) -> list[FeatureContribution]:
        return self.contributions[:n]

    def plain_language_summary(self, n: int = 4) -> list[str]:
        return [
            f"{c.display_name} ({c.value_display}) {c.direction} your risk score."
            for c in self.top(n)
        ]


def _load_bundle():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"{MODEL_PATH} not found. Run: python ml/download_diabetes_dataset.py "
            "&& python ml/train_diabetes_model.py"
        )
    return joblib.load(MODEL_PATH)


def explain_prediction(input_features: dict) -> ExplanationResult:
    bundle = _load_bundle()
    pipeline = bundle["pipeline"]

    X_background, _ = load_diabetes_data()
    background_sample = X_background.sample(
        n=min(BACKGROUND_SAMPLE_SIZE, len(X_background)), random_state=42
    )

    def predict_positive_class_proba(X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X, columns=ALL_FEATURES)
        return pipeline.predict_proba(X_df)[:, 1]

    explainer = shap.Explainer(
        predict_positive_class_proba,
        background_sample,
        feature_names=ALL_FEATURES,
    )

    X_input = pd.DataFrame([{f: input_features[f] for f in ALL_FEATURES}])
    shap_explanation = explainer(X_input)

    contributions = [
        FeatureContribution(
            feature=feature,
            display_name=DIABETES_FEATURE_DISPLAY_NAMES[feature],
            value_display=str(input_features[feature]),
            shap_value=round(float(shap_explanation.values[0][i]), 4),
        )
        for i, feature in enumerate(ALL_FEATURES)
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    base_value = float(shap_explanation.base_values[0])
    return ExplanationResult(base_value=round(base_value, 4), contributions=contributions)
