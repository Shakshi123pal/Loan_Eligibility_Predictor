import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('loan_model.pkl')

st.title("🏦 Loan Eligibility Prediction App")

# Input form
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
st.subheader("🔍 Applicant Income")

income_method = st.radio("Choose Input Method", ['Select from list', 'Enter manually'])

if income_method == 'Select from list':
    income_options = [
        '10,000', '20,000', '30,000', '40,000', '50,000',
        '75,000', '1,00,000', '1,50,000', '2,00,000', '3,00,000', '5,00,000'
    ]
    selected_income = st.selectbox("Choose Applicant Income (₹)", income_options)
    applicant_income = int(selected_income.replace(',', ''))

else:
    user_input = st.text_input("Enter Applicant Income in ₹")
    if user_input:
        try:
            applicant_income = int(user_input)
            st.success(f"Income accepted: ₹{applicant_income:,}")
        except ValueError:
            st.error("Please enter a valid number.")

coapplicant_income = st.number_input("Coapplicant Income", value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", value=100)
loan_term = st.number_input("Loan Amount Term (in months)", value=360)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# Preprocess input data
def preprocess_input():
    data = {
        'Gender': 1 if gender == 'Male' else 0,
        'Married': 1 if married == 'Yes' else 0,
        'Education': 1 if education == 'Graduate' else 0,
        'Self_Employed': 1 if self_employed == 'Yes' else 0,
        'Credit_History': int(credit_history),
        'ApplicantIncome': int(applicant_income),
        'CoapplicantIncome': int(coapplicant_income),
        'LoanAmount': int(loan_amount),
        'Loan_Amount_Term': int(loan_term),
        'Dependents': dependents,
        'Property_Area': property_area
    }

    df = pd.DataFrame([data])

    # Transform values
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncome_log'] = np.log1p(df['ApplicantIncome'])
    df['CoapplicantIncome_log'] = np.log1p(df['CoapplicantIncome'])
    df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
    df['Total_Income_log'] = np.log1p(df['Total_Income'])
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 1000)

    # Drop raw columns
    df.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income'], axis=1, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Property_Area', 'Dependents'], drop_first=True)

    # Ensure dummy columns are boolean to match training
    for col in ['Property_Area_Semiurban', 'Property_Area_Urban',
                'Dependents_1', 'Dependents_2', 'Dependents_3']:
        if col not in df.columns:
            df[col] = False
        else:
            df[col] = df[col].astype(bool)

    # Expected column order
    expected_cols = [
        'Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Amount_Term',
        'Credit_History', 'ApplicantIncome_log', 'CoapplicantIncome_log',
        'LoanAmount_log', 'Property_Area_Semiurban', 'Property_Area_Urban',
        'Dependents_1', 'Dependents_2', 'Dependents_3',
        'Total_Income_log', 'EMI', 'Balance_Income'
    ]

    # Add any missing expected columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[expected_cols]

    return df

# Predict button
if st.button("Predict Loan Eligibility"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)
    result = "✅ Eligible for Loan" if prediction[0] == 1 else "❌ Not Eligible for Loan"
    st.subheader(result)


