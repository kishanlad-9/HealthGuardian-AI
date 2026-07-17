"""
Human-readable display names for features and their categorical values.

Centralized here (not duplicated in pages/2_Heart_Disease_Prediction.py and
ml/explain_heart.py separately) so the prediction form and the SHAP
explanation always describe a feature the same way - e.g. "Chest pain type"
everywhere, never "cp" in one place and "Chest Pain" in another.
"""

FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG results",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs. rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels colored by fluoroscopy",
    "thal": "Thalassemia",
}

CP_OPTIONS = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}
RESTECG_OPTIONS = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
SLOPE_OPTIONS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_OPTIONS = {0: "Unknown/other", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}
SEX_OPTIONS = {1: "Male", 0: "Female"}
YES_NO_OPTIONS = {1: "Yes", 0: "No"}

CATEGORY_VALUE_LABELS = {
    "sex": SEX_OPTIONS,
    "cp": CP_OPTIONS,
    "fbs": YES_NO_OPTIONS,
    "restecg": RESTECG_OPTIONS,
    "exang": YES_NO_OPTIONS,
    "slope": SLOPE_OPTIONS,
    "thal": THAL_OPTIONS,
    # "ca" (count of vessels, 0-4) has no separate label map - the raw
    # number is already meaningful to display as-is.
}


def format_feature_value(feature: str, value) -> str:
    """Returns a human-readable rendering of a feature's value, e.g.
    format_feature_value("cp", 2) -> "Non-anginal pain"."""
    if feature in CATEGORY_VALUE_LABELS:
        return CATEGORY_VALUE_LABELS[feature].get(value, str(value))
    return str(value)
