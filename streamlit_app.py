import streamlit as st
import pandas as pd
import pickle

# Load data and model
car = pd.read_csv("Cleaned Car.csv")
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

st.set_page_config(page_title="Car Price Predictor")
st.title("🚗 Car Price Predictor")

# Dropdown values
companies = sorted(car["company"].unique())
years = sorted(car["year"].unique(), reverse=True)
fuel_types = sorted(car["fuel_type"].unique())

# Company selection
company = st.selectbox("Company", companies)

# Show only models of the selected company
filtered_cars = sorted(
    car[car["company"] == company]["name"].unique()
)

# Car model selection
car_name = st.selectbox("Car Name", filtered_cars)

# Other inputs
year = st.selectbox("Year", years)
fuel = st.selectbox("Fuel Type", fuel_types)
kms = st.number_input("Kilometers Driven", min_value=0)

if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[car_name, company, year, kms, fuel]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    prediction = model.predict(input_df)

    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
