import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        return pickle.load(f)

# ------------------ Load Dataset ------------------
@st.cache_resource
def load_data():
    return pd.read_csv("data.csv")   # <-- your dataset name

model = load_model()
data = load_data()

st.title("📘 Student Prediction Model – Streamlit App")

st.write("This app uses your trained ML model to make predictions.")

# ------------------ Dynamically Build Inputs ------------------
st.subheader("Enter Input Values")

input_values = {}

for col in data.columns:
    if col == data.columns[-1]:  
        continue  

    if pd.api.types.is_numeric_dtype(data[col]):
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        default_val = float(data[col].median())
        input_values[col] = st.number_input(
            f"{col}", min_value=min_val, max_value=max_val, value=default_val
        )
    else:
        unique_vals = data[col].dropna().unique().tolist()
        input_values[col] = st.selectbox(f"{col}", unique_vals)

# Convert to DataFrame
input_df = pd.DataFrame([input_values])

# ------------------ Predict ------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
