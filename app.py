import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Telco Customer Churn Predictor")
st.caption("RandomForest | ROC-AUC 0.839 | Best params: n=200, depth=10")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 24)
    monthly = st.number_input("MonthlyCharges", 18.0, 120.0, 70.0)
    total = st.number_input("TotalCharges", 0.0, 9000.0, tenure*monthly)
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

with col2:
    online_sec = st.selectbox(
        "OnlineSecurity", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox(
        "TechSupport", ["Yes", "No", "No internet service"])
    payment = st.selectbox("PaymentMethod", [
                           "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])

data = {
    'gender': 'Male',
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': 'No',
    'tenure': tenure,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': internet,
    'OnlineSecurity': online_sec,
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': tech_support,
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': contract,
    'PaperlessBilling': 'Yes',
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': total
}

if st.button("Predict Churn"):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    st.metric("Churn Probability", f"{prob:.1%}")

    if pred == 1:
        st.error(
            f"🔴 High Risk - {prob:.0%} will churn. Offer retention / discount.")
    else:
        st.success(
            f"🟢 Low Risk - {prob:.0%} churn risk. Customer likely to stay.")
