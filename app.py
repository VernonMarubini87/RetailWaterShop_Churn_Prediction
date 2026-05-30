# Gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'


import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button to see the result")

st.divider()

age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=12)

monthly_charges = st.number_input("Enter Monthly Charges", min_value=30, max_value=1000, value=150) 

gender = st.selectbox("Enter the Gender",["Male","Female"])

st.divider()

predict_button = st.button("Predict!")

st.divider()

if predict_button:

    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthly_charges]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()

    st.write(f"Predicted: {predicted}")

    if prediction == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")

else:    
    st.write("Please enter the values and hit the predict button to see the result")   


