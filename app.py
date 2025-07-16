# Import the required libraries
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import gzip
import os
import urllib.request

# Set the page configuration of the app
st.set_page_config(
    page_title="Timelytics: OTD Prediction",
    page_icon=":pencil:",
    layout="wide",
)

# Display the title and captions for the app
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - "
    "XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to "
    "Delivery (OTD) times."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain "
    "and take proactive measures to address them, reducing lead times and improving delivery times."
)

# Load the trained ensemble model (from compressed .pkl.gz file)
@st.cache_resource
def load_model():
    model_path = "voting_model.pkl.gz"
    
    # Optional: Download from external source if not included in repo
    # url = "https://your-link-to-release-or-drive/voting_model.pkl.gz"
    # if not os.path.exists(model_path):
    #     urllib.request.urlretrieve(url, model_path)

    with gzip.open(model_path, "rb") as file:
        return pickle.load(file)

voting_model = load_model()

# Prediction function
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    prediction = voting_model.predict(
        np.array(
            [
                [
                    purchase_dow,
                    purchase_month,
                    year,
                    product_size_cm3,
                    product_weight_g,
                    geolocation_state_customer,
                    geolocation_state_seller,
                    distance,
                ]
            ]
        )
    )
    return round(prediction[0])

# Sidebar for input
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")

    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", min_value=2000, max_value=2100, value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", min_value=1, value=9328)
    product_weight_g = st.number_input("Product Weight in grams", min_value=1, value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", min_value=0, value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", min_value=0, value=20)
    distance = st.number_input("Distance (in km)", min_value=0.0, value=475.35)

    submit = st.button("Predict OTD Time")

# Output container
with st.container():
    st.header("Output: Predicted Wait Time (in Days)")
    if submit:
        with st.spinner("This may take a moment..."):
            prediction = waitime_predictor(
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            )
            st.success(f"📦 Estimated Delivery Time: **{prediction} days**")

    # Display sample dataset
    sample_data = {
        "Purchased Day of the Week": [0, 3, 1],
        "Purchased Month": [6, 3, 1],
        "Purchased Year": [2018, 2017, 2018],
        "Product Size in cm³": [37206.0, 63714, 54816],
        "Product Weight in grams": [16250.0, 7249, 9600],
        "Geolocation State Customer": [25, 25, 25],
        "Geolocation State Seller": [20, 7, 20],
        "Distance (in km)": [247.94, 250.35, 4.915],
    }

    df = pd.DataFrame(sample_data)

    st.header("📊 Sample Dataset (For Reference)")
    st.write(df)
