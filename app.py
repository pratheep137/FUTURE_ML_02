import streamlit as st
import pandas as pd
import joblib

model = joblib.load('../model/model.pkl')
st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload customer CSV", type="csv")
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    input_df_encoded = input_df.copy()  # add encoding here
    preds = model.predict(input_df_encoded)
    input_df['Churn Prediction'] = preds
    st.write(input_df)
