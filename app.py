import streamlit as st
import pandas as pd
import pickle

st.title("📘 Student Model Prediction App")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.write("Enter the feature values below:")

# ------------------ YOUR INPUT FIELDS ------------------
# ❗ Replace these with your actual feature names
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# Put all inputs into a DataFrame
input_df = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                        columns=["Feature1", "Feature2", "Feature3", "Feature4"])

# ------------------ Predict ------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
