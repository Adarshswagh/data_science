import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("logistic_model.pkl")

st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings / Spouses aboard", 0, 8, 0)
parch = st.number_input("Parents / Children aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

sex_val = 1 if sex == "male" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_df = pd.DataFrame([[
    pclass, sex_val, age, sibsp, parch, fare,
    embarked_C, embarked_Q, embarked_S
]], columns=[
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked_C", "Embarked_Q", "Embarked_S"
])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"Passenger will SURVIVE (Probability: {prob:.2f})")
    else:
        st.error(f"Passenger will NOT survive (Probability: {1 - prob:.2f})")