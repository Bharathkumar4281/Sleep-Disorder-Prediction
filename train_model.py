# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")
df.drop(columns=["Person ID"], inplace=True)

# Clean Blood Pressure values
df["Blood Pressure"] = df["Blood Pressure"].map({
    "120/80": "Normal", "125/80": "Normal", "126/83": "Normal", "128/85": "Normal",
    "135/85": "High", "140/90": "High", "130/85": "High",
    "110/70": "Low", "105/60": "Low", "100/60": "Low"
})

# Drop missing values
df.dropna(inplace=True)

# Encode categorical
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["BMI Category"] = df["BMI Category"].map({
    "Normal": 0, "Overweight": 1, "Obese": 2, "Underweight": 3
})
df["Blood Pressure"] = df["Blood Pressure"].map({
    "Normal": 0, "High": 1, "Low": 2
})

# One-hot encode occupation
df = pd.get_dummies(df, columns=["Occupation"], drop_first=True)

# Split X and y
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and feature list
joblib.dump(model, "model/sleep_disorder_model.pkl")
joblib.dump(list(X.columns), "model/model_features.pkl")

print("✅ Model and features saved successfully.")
