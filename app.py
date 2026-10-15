import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Load the saved files
model = joblib.load("car_price_model.pkl")
columns = list(model.feature_names_in_)
options = json.load(open("options.json"))

# 2. Page title
st.title("Used Car Price Estimator")

# 3. Input boxes
brand = st.selectbox("Brand", options["brand"])
fuel = st.selectbox("Fuel type", options["fuel_type"])
transmission = st.selectbox("Transmission", options["transmission_type"])
seller = st.selectbox("Seller type", options["seller_type"])
seats = st.selectbox("Seats", options["seats"])

age = st.slider("Vehicle age (years)", 0, 20, 5)
km = st.number_input("Kilometres driven", 0, 500000, 40000)
mileage = st.number_input("Mileage (km per litre)", 5.0, 35.0, 18.0)
engine = st.number_input("Engine size (cc)", 600, 6000, 1200)
power = st.number_input("Max power (bhp)", 30.0, 700.0, 85.0)

# 4. When the button is clicked, predict
if st.button("Estimate price"):
    row = pd.DataFrame([{
        "vehicle_age": age,
        "km_driven": km,
        "mileage": mileage,
        "engine": engine,
        "max_power": power,
        "seats": seats,
        "brand": brand,
        "seller_type": seller,
        "fuel_type": fuel,
        "transmission_type": transmission,
    }])

    row = pd.get_dummies(row, dtype=int)
    row = row.reindex(columns=columns, fill_value=0)

    price = np.expm1(model.predict(row)[0])
    st.success(f"Estimated price: ₹{price:,.0f}")
