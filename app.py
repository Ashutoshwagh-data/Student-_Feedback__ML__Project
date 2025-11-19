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

st.write("Enter the required feature values below:")

# Input fields with correct feature names
age = st.number_input("Age", min_value=0, max_value=100, value=20)
department = st.selectbox("Department", ["IT", "HR", "Finance", "Sales"])
experience = st.number_input("Experience (Years)", min_value=0, max_value=40, value=1)
salary = st.number_input("Salary", min_value=0, max_value=200000, value=30000)

# Create dataframe with correct column names
input_df = pd.DataFrame([{
    "Age": age,
    "Department": department,
    "Experience": experience,
    "Salary": salary
}])

# Predict
if st.button("Predict"):
    try:
        result = model.predict(input_df)[0]
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
