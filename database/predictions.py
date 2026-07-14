"""
Data access layer for prediction_history.

Design decision: this module takes/returns plain Python types (dict for
input_features, not a disease-specific class) because the whole point of
the JSON-column schema (see schema.sql) is that this layer doesn't need to
know what a "heart disease" or "diabetes" input looks like. The ml/ layer
(Milestone 4+) owns feature shape; this module just persists whatever it's
given under a disease_type tag.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from database.db import get_connection

DiseaseType = Literal["heart_disease", "diabetes"]
RiskLabel = Literal["Low", "Medium", "High"]


@dataclass(frozen=True)
class PredictionRecord:
    id: int
    user_id: int
    disease_type: str
    input_features: dict
    risk_probability: float
    risk_label: str
    model_version: str
    shap_values: dict | None
    created_at: str

    @classmethod
    def _from_row(cls, row) -> "PredictionRecord":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            disease_type=row["disease_type"],
            input_features=json.loads(row["input_features"]),
            risk_probability=row["risk_probability"],
            risk_label=row["risk_label"],
            model_version=row["model_version"],
            shap_values=json.loads(row["shap_values"]) if row["shap_values"] else None,
            created_at=row["created_at"],
        )


def save_prediction(
    user_id: int,
    disease_type: DiseaseType,
    input_features: dict,
    risk_probability: float,
    risk_label: RiskLabel,
    model_version: str,
    shap_values: dict | None = None,
) -> PredictionRecord:
    """Persist one prediction and return the saved record (with its new id)."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history
                (user_id, disease_type, input_features, risk_probability,
                 risk_label, model_version, shap_values)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                disease_type,
                json.dumps(input_features),
                risk_probability,
                risk_label,
                model_version,
                json.dumps(shap_values) if shap_values is not None else None,
            ),
        )
        conn.commit()
        return get_prediction_by_id(cursor.lastrowid)
    finally:
        conn.close()


def get_prediction_by_id(prediction_id: int) -> PredictionRecord | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM prediction_history WHERE id = ?", (prediction_id,)
        ).fetchone()
        return PredictionRecord._from_row(row) if row else None
    finally:
        conn.close()


def get_predictions_for_user(
    user_id: int, disease_type: DiseaseType | None = None, limit: int = 50
) -> list[PredictionRecord]:
    """Most recent first. Optionally filtered to one disease type (for
    per-disease dashboard tabs in later milestones)."""
    conn = get_connection()
    try:
        if disease_type:
            rows = conn.execute(
                """SELECT * FROM prediction_history
                   WHERE user_id = ? AND disease_type = ?
                   ORDER BY created_at DESC, id DESC LIMIT ?""",
                (user_id, disease_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM prediction_history
                   WHERE user_id = ?
                   ORDER BY created_at DESC, id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [PredictionRecord._from_row(row) for row in rows]
    finally:
        conn.close()


def delete_prediction(prediction_id: int, user_id: int) -> bool:
    """Delete a prediction, scoped to user_id so one user can never delete
    another's record even if they guess an id. Returns True if a row was
    actually deleted."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM prediction_history WHERE id = ? AND user_id = ?",
            (prediction_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
