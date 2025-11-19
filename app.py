import streamlit as st
import pandas as pd
import pickle

st.title("Student Model Prediction App")

# Load model
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------
# IMPORTANT:
# Set the correct order EXACTLY like your training dataset
# --------------------
correct_order = ["Age", "Department", "Experience", "Salary"]

st.write("Enter the feature values:")

age = st.number_input("Age", 0, 100, 20)
department = st.selectbox("Department", ["IT", "HR", "Finance", "Sales"])
experience = st.number_input("Experience", 0, 50, 1)
salary = st.number_input("Salary", 0, 200000, 30000)

# Create dataframe (unordered)
input_data = {
    "Age": age,
    "Department": department,
    "Experience": experience,
    "Salary": salary
}

# Reorder columns to match training order
input_df = pd.DataFrame([input_data])[correct_order]

# Predict
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"Prediction: {pred}")
    except Exception as e:
        st.error(f"Error: {e}")
