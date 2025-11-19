import streamlit as st
import pickle
import pandas as pd

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------------- UI ----------------------
st.title("Student Performance Prediction App")
st.write("Enter the details below to get the prediction.")

# Example input fields — change according to your dataset features
gender = st.selectbox("Gender", ["male", "female"])
study_hours = st.number_input("Study hours per day", min_value=0, max_value=10)
attendance = st.slider("Attendance (%)", 0, 100)
parent_education = st.selectbox("Parent Education", ["High School", "Graduate", "Post Graduate"])

# Convert categorical to numeric (modify as per your data preprocessing)
def preprocess_input():
    gender_map = {"male": 0, "female": 1}
    parent_edu_map = {"High School": 0, "Graduate": 1, "Post Graduate": 2}

    data = {
        "gender": gender_map[gender],
        "study_hours": study_hours,
        "attendance": attendance,
        "parent_education": parent_edu_map[parent_education],
    }

    return pd.DataFrame([data])

# ---------------------- Prediction ----------------------
if st.button("Predict"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Score: {prediction}")
