"""
Preprocessing for the heart disease dataset.

Design decision on feature treatment: sex, cp, fbs, restecg, exang, slope,
ca, and thal are stored as integers in the raw CSV but are NOMINAL
categories, not ordinal/continuous quantities (e.g. cp=0..3 are four
distinct chest-pain types with no inherent order - cp=2 isn't "between"
cp=1 and cp=3 in any meaningful clinical sense). One-hot encoding them
avoids the model learning a false ordinal relationship. age, trestbps,
chol, thalach, and oldpeak are genuinely continuous and get scaled instead.

Known data-quality caveat (documented, not silently fixed): in this
specific mirror of the dataset, ca=4 and thal=0 are widely reported
(see datasets/README.md) to actually represent missing-value codes from
the original UCI encoding, not real categories. Left as-is here - the
one-hot encoding treats them as their own category rather than imputing,
which is defensible for a portfolio project but worth knowing about
before trusting this model on any real clinical use case.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "heart_disease_uci.csv"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_heart_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads the dataset and splits into features (X) and target (y).

    IMPORTANT - label inversion fix: in this dataset file, the raw `target`
    column is 0 for disease-PRESENT and 1 for disease-ABSENT - the opposite
    of the intuitive convention (and the opposite of what the column name
    suggests). This is a known, documented quirk of this specific dataset
    lineage (search "kaggle ronitf heart-disease-uci discussion 105877" -
    the same reprocessed file this mirror is based on). Verified directly
    against the data before trusting the discussion: the raw target=0 group
    has higher age/cholesterol/oldpeak, lower max heart rate, ~4x the
    exercise-induced-angina rate, and ~3x the rate of blocked vessels
    (ca > 0) compared to target=1 - all classic heart-disease indicators,
    confirming target=0 is the disease-positive group.

    This function returns y INVERTED (`1 - raw_target`) so that everywhere
    else in this codebase - predict_heart.py, prediction_history.risk_label,
    the UI - `1` / "positive" consistently means "heart disease present",
    matching normal convention. Do not reintroduce the raw column directly.

    Raises FileNotFoundError with an actionable message if the dataset
    hasn't been downloaded yet.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run: python ml/download_heart_dataset.py"
        )
    df = pd.read_csv(DATASET_PATH)
    X = df[ALL_FEATURES]
    y = 1 - df[TARGET_COLUMN]  # invert - see docstring
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Returns an unfit ColumnTransformer: StandardScaler for numeric
    features, OneHotEncoder for categorical features. Fit this as part of
    a Pipeline with a classifier (see train_heart_model.py) rather than
    fitting it standalone, so the same fitted transform is guaranteed to
    apply to both training and future inference inputs."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    """Wraps a classifier with the preprocessor into a single Pipeline, so
    saved_models/*.joblib contains preprocessing + model as one artifact -
    inference code never needs to remember to scale/encode features itself."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)])
