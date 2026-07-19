"""Unit tests for database/predictions.py, using the same temp-db pattern
as test_auth.py."""
import sqlite3
from pathlib import Path

import pytest

import database.db as db_module
from authentication.auth import signup
from database.predictions import (
    delete_prediction,
    get_prediction_by_id,
    get_predictions_for_user,
    save_prediction,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    temp_db_path = tmp_path / "test_healthguardian.db"
    monkeypatch.setattr(db_module, "DB_PATH", temp_db_path)
    db_module.init_db()
    yield


@pytest.fixture
def user_id() -> int:
    return signup("patient1", "patient1@example.com", "password123").id


def test_save_and_retrieve_prediction(user_id):
    saved = save_prediction(
        user_id=user_id,
        disease_type="heart_disease",
        input_features={"age": 54, "cholesterol": 240},
        risk_probability=0.73,
        risk_label="High",
        model_version="heart_xgboost_v1",
    )
    fetched = get_prediction_by_id(saved.id)
    assert fetched is not None
    assert fetched.user_id == user_id
    assert fetched.input_features == {"age": 54, "cholesterol": 240}
    assert fetched.risk_probability == 0.73
    assert fetched.risk_label == "High"
    assert fetched.shap_values is None


def test_save_prediction_with_shap_values(user_id):
    saved = save_prediction(
        user_id=user_id,
        disease_type="diabetes",
        input_features={"glucose": 148, "bmi": 33.6},
        risk_probability=0.41,
        risk_label="Medium",
        model_version="diabetes_xgboost_v1",
        shap_values={"glucose": 0.21, "bmi": 0.08},
    )
    fetched = get_prediction_by_id(saved.id)
    assert fetched.shap_values == {"glucose": 0.21, "bmi": 0.08}


def test_get_predictions_for_user_ordered_most_recent_first(user_id):
    first = save_prediction(user_id, "heart_disease", {"age": 50}, 0.2, "Low", "v1")
    second = save_prediction(user_id, "heart_disease", {"age": 60}, 0.6, "Medium", "v1")

    history = get_predictions_for_user(user_id)
    assert len(history) == 2
    assert history[0].id == second.id  # most recent first
    assert history[1].id == first.id


def test_get_predictions_filtered_by_disease_type(user_id):
    save_prediction(user_id, "heart_disease", {"age": 50}, 0.2, "Low", "v1")
    save_prediction(user_id, "diabetes", {"glucose": 100}, 0.3, "Low", "v1")

    heart_only = get_predictions_for_user(user_id, disease_type="heart_disease")
    assert len(heart_only) == 1
    assert heart_only[0].disease_type == "heart_disease"


def test_predictions_scoped_to_user(user_id):
    other_user_id = signup("patient2", "patient2@example.com", "password123").id
    save_prediction(user_id, "heart_disease", {"age": 50}, 0.2, "Low", "v1")

    assert len(get_predictions_for_user(user_id)) == 1
    assert len(get_predictions_for_user(other_user_id)) == 0


def test_delete_prediction_removes_row(user_id):
    saved = save_prediction(user_id, "heart_disease", {"age": 50}, 0.2, "Low", "v1")
    deleted = delete_prediction(saved.id, user_id)
    assert deleted is True
    assert get_prediction_by_id(saved.id) is None


def test_delete_prediction_fails_for_wrong_user(user_id):
    other_user_id = signup("patient2", "patient2@example.com", "password123").id
    saved = save_prediction(user_id, "heart_disease", {"age": 50}, 0.2, "Low", "v1")

    deleted = delete_prediction(saved.id, other_user_id)
    assert deleted is False
    assert get_prediction_by_id(saved.id) is not None  # still there


def test_invalid_disease_type_rejected_by_schema(user_id):
    with pytest.raises(sqlite3.IntegrityError):
        save_prediction(user_id, "unknown_disease", {"age": 50}, 0.2, "Low", "v1")


def test_invalid_risk_label_rejected_by_schema(user_id):
    with pytest.raises(sqlite3.IntegrityError):
        save_prediction(user_id, "heart_disease", {"age": 50}, 0.2, "Extreme", "v1")


def test_out_of_range_probability_rejected_by_schema(user_id):
    with pytest.raises(sqlite3.IntegrityError):
        save_prediction(user_id, "heart_disease", {"age": 50}, 1.5, "High", "v1")


def test_cascade_delete_removes_predictions_when_user_deleted(user_id):
    save_prediction(user_id, "heart_disease", {"age": 50}, 0.2, "Low", "v1")
    conn = db_module.get_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    assert len(get_predictions_for_user(user_id)) == 0
