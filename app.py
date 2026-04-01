import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load model and features
model = joblib.load("model/sleep_disorder_model.pkl")
model_features = joblib.load("model/model_features.pkl")

# Page Config
st.set_page_config(page_title="Sleep Disorder Predictor", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #0e76a8;'>😴 Sleep Disorder Risk Prediction</h1>
    <p style='text-align: center;'>Enter your health and lifestyle info to check for sleep disorder risks</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar - User Input
st.sidebar.header("📝 User Input Parameters")

def user_input():
    Age = st.sidebar.slider("Age", 18, 90, 30)
    Gender = st.sidebar.radio("Gender", ["Male", "Female"])
    Occupation = st.sidebar.selectbox("Occupation", [
        "Doctor", "Engineer", "Lawyer", "Nurse", "Sales Representative",
        "Scientist", "Software Engineer", "Teacher", "Accountant"
    ])
    Sleep_Duration = st.sidebar.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, 0.1)
    Quality_of_Sleep = st.sidebar.slider("Quality of Sleep (1-10)", 1, 10, 6)
    Physical_Activity_Level = st.sidebar.slider("Physical Activity Level (0-100)", 0, 100, 50)
    Stress_Level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
    BMI_Category = st.sidebar.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    Blood_Pressure = st.sidebar.selectbox("Blood Pressure", ["Low", "Normal", "High"])
    Heart_Rate = st.sidebar.slider("Heart Rate", 40, 120, 72)
    Daily_Steps = st.sidebar.slider("Daily Steps", 1000, 20000, 7000, 1000)

    # Map input values to match training encodings
    data = {
        "Age": Age,
        "Gender": 1 if Gender == "Male" else 0,
        "Sleep Duration": Sleep_Duration,
        "Quality of Sleep": Quality_of_Sleep,
        "Physical Activity Level": Physical_Activity_Level,
        "Stress Level": Stress_Level,
        "BMI Category": {
            "Normal": 0, "Overweight": 1, "Obese": 2, "Underweight": 3
        }[BMI_Category],
        "Blood Pressure": {
            "Normal": 0, "High": 1, "Low": 2
        }[Blood_Pressure],
        "Heart Rate": Heart_Rate,
        "Daily Steps": Daily_Steps,
    }

    # One-hot encode Occupation
    for occ in ["Doctor", "Engineer", "Lawyer", "Nurse", "Sales Representative", "Scientist", "Software Engineer", "Teacher"]:
        data[f"Occupation_{occ}"] = 1 if occ == Occupation else 0

    return pd.DataFrame(data, index=[0])

# Collect input
input_df = user_input()

# Add any missing columns
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match model
input_df = input_df[model_features]

# Predict button
if st.button("🩺 Predict Sleep Disorder"):
    prediction = model.predict(input_df)[0]
    st.markdown(f"""
        <div style='text-align: center; margin-top: 40px;'>
            <h2 style='color: #1f77b4;'>🧾 Result: <span style='color: green;'>{prediction}</span></h2>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Enter the values from the sidebar and click the button to predict.")
