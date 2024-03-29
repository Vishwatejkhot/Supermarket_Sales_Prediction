import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the function to make predictions
def predict_rating(inputs):
    # Reshape inputs to ensure it's a 2D array
    inputs = np.array(inputs).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(inputs)

    return prediction[0]

# Streamlit app layout
def main():
    st.title('Supermarket Sales Rating Prediction')

    # Inputs
    branch = st.slider('Branch', 0, 2, 0)
    city = st.slider('City', 0, 2, 0)
    customer_type = st.slider('Customer Type', 0, 1, 0)
    gender = st.slider('Gender', 0, 1, 0)
    product_line = st.slider('Product Line', 0, 4, 0)
    unit_price = st.number_input('Unit Price', value=0.0)
    quantity = st.number_input('Quantity', value=0)
    tax = st.number_input('Tax (5%)', value=0.0)
    total = st.number_input('Total', value=0.0)
    payment = st.slider('Payment', 0, 2, 0)
    cogs = st.number_input('Cost of Goods Sold (COGS)', value=0.0)
    gross_margin_percentage = st.number_input('Gross Margin Percentage', value=0.0)
    gross_income = st.number_input('Gross Income', value=0.0)
    month = st.slider('Month', 1, 12, 1)
    year = st.slider('Year', 2019, 2024, 2019)
    hour = st.slider('Hour', 0, 23, 0)

    # Predict rating
    inputs = [
        branch, city, customer_type, gender, product_line,
        unit_price, quantity, tax, total, payment,
        cogs, gross_margin_percentage, gross_income,
        month, year, hour
    ]

    if st.button('Predict Rating'):
        rating = predict_rating(inputs)
        st.write(f'Predicted Rating: {rating}')

if __name__ == '__main__':
    main()
