"""
Preprocessing for the Pima Indians Diabetes dataset.

Data quality issue handled here (verified empirically before writing this,
not assumed from documentation alone - see ml/download_diabetes_dataset.py
and the training report for the actual zero-counts found): glucose,
blood_pressure, skin_thickness, insulin, and bmi use 0 as a missing-value
placeholder - a live patient cannot have 0 blood pressure or 0 BMI. Left
as literal zeros, these would corrupt the model (e.g. "0 BMI" would look
like an extreme, presumably healthy, outlier rather than missing data).

`pregnancies` is NOT included in that treatment - 0 pregnancies is a
completely legitimate value (and duly common), not a missing-data flag.

Fix: those five columns are converted to NaN wherever they're 0, then
imputed with SimpleImputer(strategy="median") INSIDE the pipeline, fit
on the training fold only - never using test-set values to fill training
gaps, and never imputing before the train/test split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "pima_diabetes.csv"

ALL_FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]
TARGET_COLUMN = "outcome"

# Columns where 0 is a missing-value placeholder, not a real measurement.
ZERO_AS_MISSING_COLUMNS = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def load_diabetes_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads the dataset and splits into features (X) and target (y).
    Converts biologically-impossible zeros to NaN in the affected columns
    (imputation itself happens later, inside build_pipeline(), to avoid
    leaking test-set statistics into training data).
    Raises FileNotFoundError with an actionable message if not downloaded."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run: python ml/download_diabetes_dataset.py"
        )
    df = pd.read_csv(DATASET_PATH)
    X = df[ALL_FEATURES].copy()
    X[ZERO_AS_MISSING_COLUMNS] = X[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    y = df[TARGET_COLUMN]
    return X, y


def build_pipeline(classifier) -> Pipeline:
    """Median imputation (fit on train fold only, via Pipeline) + scaling,
    then the classifier - all as one artifact, same pattern as
    ml/preprocessing.py's heart-disease pipeline."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
