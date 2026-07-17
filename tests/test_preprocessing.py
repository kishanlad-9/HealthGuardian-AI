"""Unit tests for ml/preprocessing.py. Uses the real dataset (small, 303
rows, downloaded via ml/download_heart_dataset.py) rather than a mock -
worth catching real schema drift if the dataset file ever changes."""
import pytest

from ml.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    build_preprocessor,
    load_heart_data,
)


@pytest.fixture(scope="module")
def heart_data():
    return load_heart_data()


def test_load_heart_data_shape(heart_data):
    X, y = heart_data
    assert len(X) == len(y)
    assert len(X) > 0
    assert set(X.columns) == set(ALL_FEATURES)


def test_load_heart_data_target_is_binary(heart_data):
    _, y = heart_data
    assert set(y.unique()) <= {0, 1}


def test_no_missing_values(heart_data):
    X, y = heart_data
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_feature_lists_are_disjoint_and_complete():
    assert set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES) == set()
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(ALL_FEATURES)


def test_target_direction_matches_known_risk_factors(heart_data):
    """Regression guard for a real bug: this dataset's raw `target` column
    is inverted (0=disease-present, 1=disease-absent - see load_heart_data
    docstring). If this ever silently regresses (e.g. someone "fixes" the
    inversion thinking it's a leftover mistake), predictions would flip to
    calling healthy profiles high-risk. Checks two independent, well-known
    heart-disease indicators point the right direction after the fix:
    higher age and lower max heart rate should correlate with y=1 (disease)."""
    X, y = heart_data
    disease_group_age = X.loc[y == 1, "age"].mean()
    healthy_group_age = X.loc[y == 0, "age"].mean()
    assert disease_group_age > healthy_group_age, (
        "Disease-positive group should skew older - if this fails, the "
        "target inversion fix in load_heart_data() may have regressed."
    )

    disease_group_thalach = X.loc[y == 1, "thalach"].mean()
    healthy_group_thalach = X.loc[y == 0, "thalach"].mean()
    assert disease_group_thalach < healthy_group_thalach, (
        "Disease-positive group should have LOWER max heart rate achieved "
        "(thalach) - if this fails, check the target inversion in "
        "load_heart_data()."
    )


def test_preprocessor_transforms_without_error(heart_data):
    X, _ = heart_data
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    # one-hot encoding expands columns - transformed should have more
    # columns than the original numeric+categorical count
    assert transformed.shape[1] > len(ALL_FEATURES)


def test_pipeline_fits_and_predicts(heart_data):
    from sklearn.linear_model import LogisticRegression

    X, y = heart_data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}
    assert probabilities.shape == (len(X), 2)


def test_missing_dataset_raises_actionable_error(tmp_path, monkeypatch):
    import ml.preprocessing as preprocessing_module

    monkeypatch.setattr(preprocessing_module, "DATASET_PATH", tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError, match="download_heart_dataset"):
        preprocessing_module.load_heart_data()
