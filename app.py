import streamlit as st
import numpy as np
import pickle
import time
import pandas as pd

# ---------------------------------------------------
# Sidebar Info
# ---------------------------------------------------
st.sidebar.caption("Developed by Rimsha Taslim")

# ---------------------------------------------------
# Header Section
# ---------------------------------------------------
st.header("Heartlytics AI")
st.caption("Machine Learning–Based Clinical Risk Prediction System")
st.caption("Developed by Rimsha Taslim")

data = """
Cardiovascular diseases remain one of the leading causes of death worldwide.
Early identification of potential heart conditions can greatly improve
treatment outcomes and patient survival rates.

This application demonstrates how machine learning can assist healthcare
professionals by predicting the likelihood of heart disease based on
medical attributes.
"""

st.markdown(data)

st.image(
    "https://i0.wp.com/asianheartinstitute.org/wp-content/uploads/2024/11/Understanding-How-Heart-Disease-Impacts-Your-Body.jpg?fit=1572%2C917&ssl=1"
)

st.divider()

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------
st.sidebar.header("Enter Patient Data")

age = st.sidebar.number_input("Age", min_value=29, max_value=77, value=45)

sex = st.sidebar.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3]
)

trestbps = st.sidebar.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=94,
    max_value=200,
    value=120
)

chol = st.sidebar.number_input(
    "Cholesterol (mg/dl)",
    min_value=126,
    max_value=564,
    value=200
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.sidebar.selectbox(
    "Resting ECG Results",
    options=[0, 1, 2]
)

thalach = st.sidebar.number_input(
    "Maximum Heart Rate Achieved",
    min_value=71,
    max_value=202,
    value=150
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.sidebar.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=6.2,
    value=1.0,
    step=0.1
)

slope = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[0, 1, 2]
)

ca = st.sidebar.selectbox(
    "Number of Major Vessels (0–4)",
    options=[0, 1, 2, 3, 4]
)

thal = st.sidebar.selectbox(
    "Thalassemia",
    options=[0, 1, 2, 3]
)

# Final Input Array (Correct Feature Order)
final_value = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

st.divider()

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
if st.button("Predict Heart Disease Risk"):

    progress_bar = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    prediction = model.predict(final_value)[0]

    # Show probability if supported
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(final_value)[0][0]
        st.info(f"Predicted Risk Probability: {probability*100:.2f}%")

    if prediction == 1:
        st.success("Low Risk of Heart Disease ✅")
    else:
        st.error("High Risk of Heart Disease ⚠️")

st.divider()

st.info("Disclaimer: This tool is for educational purposes only and not a medical diagnosis.")