"""
SHAP-based explanations for individual heart disease predictions.

Design decisions:
- shap.TreeExplainer (not KernelExplainer): the winning model is a
  RandomForestClassifier (see saved_models/README.md), and TreeExplainer
  computes EXACT SHAP values for tree ensembles in polynomial time, versus
  KernelExplainer's model-agnostic but approximate, much slower sampling
  approach. If a future retrain picks a non-tree model, this module would
  need a different explainer (shap.LinearExplainer for logistic
  regression, etc.) - it's coupled to the model type on purpose, not
  hidden behind a false abstraction.
- One-hot columns are aggregated back to their original feature: SHAP
  naturally operates on the model's actual input space (26 columns after
  one-hot encoding), but "cp_2 contributed +0.08" means nothing to a user
  who only entered one "chest pain type" value. Contributions for all
  dummy columns belonging to one source feature are summed, so the user
  sees one number per feature they actually filled in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

import joblib
import pandas as pd
import shap

from ml.feature_labels import FEATURE_DISPLAY_NAMES, format_feature_value
from ml.preprocessing import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "heart_disease_model.joblib"


class ModelNotTrainedError(Exception):
    pass


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    display_name: str
    value_display: str
    shap_value: float  # positive = pushed prediction toward "disease present"

    @property
    def direction(self) -> str:
        return "increased" if self.shap_value > 0 else "decreased"


@dataclass(frozen=True)
class ExplanationResult:
    base_value: float  # average model output over training data, before seeing this patient's features
    contributions: list[FeatureContribution]  # sorted by |shap_value| descending

    def top(self, n: int = 5) -> list[FeatureContribution]:
        return self.contributions[:n]

    def plain_language_summary(self, n: int = 4) -> list[str]:
        """Returns up to n plain-English sentences for the top contributing
        factors, e.g. 'Thalassemia (Reversible defect) increased your risk
        score.' Ordered by impact magnitude, same as self.top()."""
        return [
            f"{c.display_name} ({c.value_display}) {c.direction} your risk score."
            for c in self.top(n)
        ]


def _transformed_column_to_source_feature(column_name: str) -> str:
    """'numeric__age' -> 'age'; 'categorical__cp_2' -> 'cp'. Matches
    against known feature names rather than naive string splitting, so a
    feature name that itself contains an underscore wouldn't break this."""
    if column_name.startswith("numeric__"):
        return column_name.removeprefix("numeric__")
    remainder = column_name.removeprefix("categorical__")
    # Try the longest matching known categorical feature name as a prefix
    # (longest-first avoids a short name like "ca" wrongly matching before "cp" is checked, etc.)
    for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if remainder == feature or remainder.startswith(f"{feature}_"):
            return feature
    raise ValueError(f"Could not map transformed column '{column_name}' back to a source feature")


def _load_bundle():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"{MODEL_PATH} not found. Run: python ml/download_heart_dataset.py "
            "&& python ml/train_heart_model.py"
        )
    return joblib.load(MODEL_PATH)


def explain_prediction(input_features: dict) -> ExplanationResult:
    """Returns per-feature SHAP contributions for one patient's prediction,
    aggregated to the original (pre-one-hot) feature space and sorted by
    impact magnitude. Positive shap_value = pushed the prediction toward
    higher disease risk; negative = pushed toward lower risk."""
    bundle = _load_bundle()
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X = pd.DataFrame([{f: input_features[f] for f in ALL_FEATURES}])
    X_transformed = preprocessor.transform(X)
    transformed_columns = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(classifier)
    shap_output = explainer.shap_values(X_transformed)

    # shap_values shape varies by SHAP version for binary classifiers:
    # either a list [class_0_array, class_1_array] or a single 3D array
    # (n_samples, n_features, n_classes). Handle both explicitly rather
    # than assuming one - this exact ambiguity is a well-known SHAP
    # version gotcha.
    if isinstance(shap_output, list):
        class_1_shap = shap_output[1][0]
        base_value = explainer.expected_value[1]
    elif shap_output.ndim == 3:
        class_1_shap = shap_output[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        class_1_shap = shap_output[0]
        base_value = explainer.expected_value

    # Aggregate one-hot columns back to source features
    per_feature_shap: dict[str, float] = {f: 0.0 for f in ALL_FEATURES}
    for column_name, shap_val in zip(transformed_columns, class_1_shap):
        source_feature = _transformed_column_to_source_feature(column_name)
        per_feature_shap[source_feature] += float(shap_val)

    contributions = [
        FeatureContribution(
            feature=feature,
            display_name=FEATURE_DISPLAY_NAMES[feature],
            value_display=format_feature_value(feature, input_features[feature]),
            shap_value=round(shap_value, 4),
        )
        for feature, shap_value in per_feature_shap.items()
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return ExplanationResult(base_value=round(float(base_value), 4), contributions=contributions)
