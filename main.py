import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# App title
st.title("💡 Insurance Charges Prediction App")
st.write("Enter Age, BMI, and Smoking Status to predict insurance charges.")

# Sidebar input
st.sidebar.header("Input Features")

def user_input():
    age = st.sidebar.slider("Age", 18, 100, 30)
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    smoker = st.sidebar.selectbox("Smoker", ("yes", "no"))
    smoker_yes = 1 if smoker == "yes" else 0

    # Keep only the features your model expects
    data = {
        "age": age,
        "bmi": bmi,
        "smoker_yes": smoker_yes
    }
    return pd.DataFrame(data, index=[0])

# Collect user input
input_df = user_input()

st.subheader("User Input")
st.write(input_df)

# Prediction
if st.button("Predict Charges"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Insurance Charges: ${prediction[0]:,.2f}")
