"""Diabetes risk prediction page. Same structure as
pages/2_Heart_Disease_Prediction.py - login-gated form, prediction, SHAP
explanation, history."""
import plotly.graph_objects as go
import streamlit as st

from authentication.session import require_login
from database.predictions import get_predictions_for_user, save_prediction
from ml.explain_diabetes import ModelNotTrainedError as ExplainerNotTrainedError
from ml.explain_diabetes import explain_prediction
from ml.predict_diabetes import ModelNotTrainedError, predict_diabetes

st.set_page_config(page_title="Diabetes Risk - HealthGuardian AI", page_icon="🩸")

user = require_login()

st.title("🩸 Diabetes Risk Prediction")
st.caption(
    "⚠️ This is a portfolio ML project, not a medical device. Results are not a "
    "diagnosis - always consult a healthcare professional for medical concerns. "
    "Model trained on the Pima Indians Diabetes dataset (adult women, 21+)."
)

with st.form("diabetes_prediction_form"):
    st.subheader("Patient information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Plasma glucose concentration (mg/dl)", min_value=0, max_value=300, value=110)
        blood_pressure = st.number_input("Diastolic blood pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Triceps skin fold thickness (mm)", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("2-hour serum insulin (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("Body mass index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input(
            "Diabetes pedigree function (family history score)",
            min_value=0.0, max_value=3.0, value=0.4, step=0.01,
            help="A score summarizing diabetes history in relatives - higher means stronger family history.",
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=35)

    st.caption(
        "Enter 0 for any measurement you don't know (e.g. skin thickness, insulin) - "
        "the model treats 0 in these fields as 'not measured' and estimates from typical values."
    )
    submitted = st.form_submit_button("Predict risk", type="primary")


def render_shap_chart(contributions) -> go.Figure:
    top = list(reversed(contributions))
    labels = [f"{c.display_name}" for c in top]
    values = [c.shap_value for c in top]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        xaxis_title="Impact on risk score (SHAP value)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    return fig


if submitted:
    input_features = {
        "pregnancies": pregnancies, "glucose": glucose, "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness, "insulin": insulin, "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree, "age": age,
    }
    try:
        result = predict_diabetes(input_features)
    except ModelNotTrainedError as e:
        st.error(str(e))
    else:
        label_color = {"Low": "green", "Medium": "orange", "High": "red"}[result.risk_label]
        st.markdown(f"## Risk level: :{label_color}[{result.risk_label}]")
        st.metric("Predicted probability of diabetes", f"{result.risk_probability * 100:.1f}%")
        st.progress(result.risk_probability)

        shap_values_to_save = None
        try:
            explanation = explain_prediction(input_features)
        except ExplainerNotTrainedError:
            explanation = None

        if explanation:
            st.subheader("Why this result?")
            st.caption(
                "Each factor below is compared against this model's average patient. "
                "Red pushes the score up, green pushes it down."
            )
            for sentence in explanation.plain_language_summary(n=4):
                st.markdown(f"- {sentence}")

            with st.expander("See all 8 factors"):
                st.plotly_chart(render_shap_chart(explanation.contributions), use_container_width=True)

            shap_values_to_save = {
                "base_value": explanation.base_value,
                "contributions": {c.feature: c.shap_value for c in explanation.contributions},
            }

        save_prediction(
            user_id=user.id,
            disease_type="diabetes",
            input_features=input_features,
            risk_probability=result.risk_probability,
            risk_label=result.risk_label,
            model_version=result.model_version,
            shap_values=shap_values_to_save,
        )
        st.success("Saved to your prediction history.")

st.divider()
st.subheader("Your recent diabetes predictions")
history = get_predictions_for_user(user.id, disease_type="diabetes", limit=10)
if not history:
    st.info("No predictions yet - fill out the form above to get started.")
else:
    for record in history:
        label_color = {"Low": "green", "Medium": "orange", "High": "red"}[record.risk_label]
        st.write(
            f"**{record.created_at}** — :{label_color}[{record.risk_label}] "
            f"({record.risk_probability * 100:.1f}%) — model: `{record.model_version}`"
        )
