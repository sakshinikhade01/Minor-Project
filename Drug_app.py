import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("drug-recommender.pkl")

# Title
st.title("💊 Drug Recommendation System")

# Sidebar input
st.sidebar.header("Input Patient Details")

def user_input():
    age = st.sidebar.slider('Age', 0, 100, 30)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    bp = st.sidebar.selectbox('Blood Pressure', ['LOW', 'NORMAL'])  # Only trained on these two
    cholesterol = st.sidebar.selectbox('Cholesterol', ['NORMAL', 'HIGH'])
    na_to_k = st.sidebar.slider('Na_to_K Ratio', 0.0, 40.0, 15.0)

    # Encoding inputs
    sex_val = 1 if sex == 'Male' else 0
    chol_val = 0 if cholesterol == 'NORMAL' else 1

    # One-hot encoding for BP (only LOW and NORMAL)
    bp_low = 1 if bp == 'LOW' else 0
    bp_normal = 1 if bp == 'NORMAL' else 0

    data = {
        'Age': age,
        'Sex': sex_val,
        'Cholesterol': chol_val,
        'Na_to_K': na_to_k,
        'BP_LOW': bp_low,
        'BP_NORMAL': bp_normal
    }

    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Display inputs
st.subheader("Patient Input")
st.write(input_df)

# Predict
prediction = model.predict(input_df)

# Show result
st.subheader("Recommended Drug")
st.success(f"The recommended drug is: **{prediction[0]}**")



