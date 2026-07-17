"""Heart disease risk prediction page. Requires login (require_login stops
the page here if nobody's signed in)."""
import streamlit as st

from authentication.session import require_login
from database.predictions import get_predictions_for_user, save_prediction
from ml.predict_heart import ModelNotTrainedError, predict_heart_disease

st.set_page_config(page_title="Heart Disease Risk - HealthGuardian AI", page_icon="🫀")

user = require_login()

st.title("🫀 Heart Disease Risk Prediction")
st.caption(
    "⚠️ This is a portfolio ML project, not a medical device. Results are not a "
    "diagnosis - always consult a healthcare professional for medical concerns."
)

CP_OPTIONS = {
    0: "Typical angina", 1: "Atypical angina",
    2: "Non-anginal pain", 3: "Asymptomatic",
}
RESTECG_OPTIONS = {
    0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy",
}
SLOPE_OPTIONS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_OPTIONS = {0: "Unknown/other", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}

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

        save_prediction(
            user_id=user.id,
            disease_type="heart_disease",
            input_features=input_features,
            risk_probability=result.risk_probability,
            risk_label=result.risk_label,
            model_version=result.model_version,
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
