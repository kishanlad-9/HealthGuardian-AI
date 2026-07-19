"""Unit tests for ml/preprocessing_diabetes.py."""
import numpy as np
import pytest

from ml.preprocessing_diabetes import (
    ALL_FEATURES,
    ZERO_AS_MISSING_COLUMNS,
    build_pipeline,
    load_diabetes_data,
)


@pytest.fixture(scope="module")
def diabetes_data():
    return load_diabetes_data()


def test_load_diabetes_data_shape(diabetes_data):
    X, y = diabetes_data
    assert len(X) == len(y) == 768
    assert set(X.columns) == set(ALL_FEATURES)


def test_target_is_binary(diabetes_data):
    _, y = diabetes_data
    assert set(y.unique()) <= {0, 1}


def test_zero_as_missing_columns_converted_to_nan(diabetes_data):
    """Regression guard: glucose/blood_pressure/skin_thickness/insulin/bmi
    should have NaN wherever the raw data had 0 (biologically impossible
    for a living patient) - if this ever regresses, the model would train
    on impossible zero values again."""
    X, _ = diabetes_data
    for col in ZERO_AS_MISSING_COLUMNS:
        assert X[col].isna().sum() > 0, f"{col} should have some NaN (converted from 0)"
        assert (X[col] == 0).sum() == 0, f"{col} should have no literal zeros left"


def test_pregnancies_zero_is_preserved_not_treated_as_missing(diabetes_data):
    """0 pregnancies is a legitimate value, unlike the other 5 columns -
    this guards against someone over-applying the zero-as-missing fix to
    every column."""
    X, _ = diabetes_data
    assert (X["pregnancies"] == 0).sum() > 0


def test_pipeline_handles_missing_values_and_predicts(diabetes_data):
    from sklearn.linear_model import LogisticRegression

    X, y = diabetes_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)  # would raise if NaN handling were broken
    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}


def test_missing_dataset_raises_actionable_error(tmp_path, monkeypatch):
    import ml.preprocessing_diabetes as module

    monkeypatch.setattr(module, "DATASET_PATH", tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError, match="download_diabetes_dataset"):
        module.load_diabetes_data()
