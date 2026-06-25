import pickle

import pandas as pd
import streamlit as st

# Load data
car = pd.read_csv("Cleaned Car.csv")

# Load trained pipeline
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Car Price Predictor")

companies = sorted(car["company"].unique())
car_names = sorted(car["name"].unique())
years = sorted(car["year"].unique(), reverse=True)
fuel_types = sorted(car["fuel_type"].unique())

company = st.selectbox("Company", companies)
car_name = st.selectbox("Car Name", car_names)
year = st.selectbox("Year", years)
fuel = st.selectbox("Fuel Type", fuel_types)
kms = st.number_input("Kilometers Driven", min_value=0)

if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[car_name, company, year, kms, fuel]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    prediction = model.predict(input_df)

    st.success(f"Estimated Price: ₹ {prediction[0]:,.0f}")