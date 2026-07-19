"""Heart disease risk prediction page. Requires login (require_login stops
the page here if nobody's signed in)."""
import plotly.graph_objects as go
import streamlit as st

from authentication.session import require_login
from database.predictions import get_predictions_for_user, save_prediction
from ml.explain_heart import ModelNotTrainedError as ExplainerNotTrainedError
from ml.explain_heart import explain_prediction
from ml.feature_labels import CP_OPTIONS, RESTECG_OPTIONS, SLOPE_OPTIONS, THAL_OPTIONS
from ml.predict_heart import ModelNotTrainedError, predict_heart_disease

st.set_page_config(page_title="Heart Disease Risk - HealthGuardian AI", page_icon="🫀")

user = require_login()

st.title("🫀 Heart Disease Risk Prediction")
st.caption(
    "⚠️ This is a portfolio ML project, not a medical device. Results are not a "
    "diagnosis - always consult a healthcare professional for medical concerns."
)

with st.form("heart_prediction_form"):
    st.subheader("Patient information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.radio("Sex", options=[1, 0], format_func=lambda v: "Male" if v == 1 else "Female", horizontal=True)
        cp = st.selectbox("Chest pain type", options=list(CP_OPTIONS.keys()), format_func=lambda v: CP_OPTIONS[v])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=60, max_value=240, value=120)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.radio("Fasting blood sugar > 120 mg/dl?", options=[1, 0], format_func=lambda v: "Yes" if v == 1 else "No", horizontal=True)
        restecg = st.selectbox("Resting ECG results", options=list(RESTECG_OPTIONS.keys()), format_func=lambda v: RESTECG_OPTIONS[v])

    with col2:
        thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=220, value=150)
        exang = st.radio("Exercise-induced angina?", options=[1, 0], format_func=lambda v: "Yes" if v == 1 else "No", horizontal=True)
        oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", options=list(SLOPE_OPTIONS.keys()), format_func=lambda v: SLOPE_OPTIONS[v])
        ca = st.selectbox("Number of major vessels colored by fluoroscopy", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", options=list(THAL_OPTIONS.keys()), format_func=lambda v: THAL_OPTIONS[v])

    submitted = st.form_submit_button("Predict risk", type="primary")


def render_shap_chart(contributions, n: int = 8) -> go.Figure:
    """Horizontal bar chart of the top-n SHAP contributions, red=increases
    risk, green=decreases risk - deliberately matching the Low/Medium/High
    color convention used elsewhere on this page."""
    top = contributions[:n]
    top = list(reversed(top))  # so the biggest contributor plots at the top
    labels = [f"{c.display_name} ({c.value_display})" for c in top]
    values = [c.shap_value for c in top]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        xaxis_title="Impact on risk score (SHAP value)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
    )
    return fig


if submitted:
    input_features = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    try:
        result = predict_heart_disease(input_features)
    except ModelNotTrainedError as e:
        st.error(str(e))
    else:
        label_color = {"Low": "green", "Medium": "orange", "High": "red"}[result.risk_label]
        st.markdown(f"## Risk level: :{label_color}[{result.risk_label}]")
        st.metric("Predicted probability of heart disease", f"{result.risk_probability * 100:.1f}%")
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

            with st.expander("See all 13 factors"):
                st.plotly_chart(render_shap_chart(explanation.contributions), use_container_width=True)

            shap_values_to_save = {
                "base_value": explanation.base_value,
                "contributions": {c.feature: c.shap_value for c in explanation.contributions},
            }

        save_prediction(
            user_id=user.id,
            disease_type="heart_disease",
            input_features=input_features,
            risk_probability=result.risk_probability,
            risk_label=result.risk_label,
            model_version=result.model_version,
            shap_values=shap_values_to_save,
        )
        st.success("Saved to your prediction history.")

st.divider()
st.subheader("Your recent heart disease predictions")
history = get_predictions_for_user(user.id, disease_type="heart_disease", limit=10)
if not history:
    st.info("No predictions yet - fill out the form above to get started.")
else:
    for record in history:
        label_color = {"Low": "green", "Medium": "orange", "High": "red"}[record.risk_label]
        st.write(
            f"**{record.created_at}** — :{label_color}[{record.risk_label}] "
            f"({record.risk_probability * 100:.1f}%) — model: `{record.model_version}`"
        )
