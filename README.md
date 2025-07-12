# Timelytics: Order to Delivery (OTD) Prediction

## Overview

Timelytics is an advanced machine learning-based model designed to optimize supply chain management by predicting Order to Delivery (OTD) times. The model leverages powerful algorithms such as **XGBoost**, **Random Forests**, and **Support Vector Machines (SVM)** to forecast delivery times based on several input parameters. 

Businesses can use Timelytics to identify potential bottlenecks and delays in their supply chains and take proactive measures to improve efficiency, reduce lead times, and enhance customer satisfaction.

## Key Features

- Predicts delivery times (in days) based on input features.
- Provides estimations for delivery time considering factors such as the product size, weight, and geographical locations of the customer and seller.
- Visualizes predictions and sample datasets for better understanding.
  
## Input Parameters

To make predictions, the following input parameters are used:

- **Purchased Day of the Week**: The day of the week when the order was placed (e.g., Monday = 0, Sunday = 6).
- **Purchased Month**: The month the order was placed (1 to 12).
- **Purchased Year**: The year the order was placed (e.g., 2018).
- **Product Size in cm³**: The size of the product in cubic centimeters.
- **Product Weight in grams**: The weight of the product in grams.
- **Geolocation State of the Customer**: The state or region of the customer.
- **Geolocation State of the Seller**: The state or region of the seller.
- **Distance (in km)**: The distance between the customer and seller (in kilometers).

## Installation

### Prerequisites

Ensure you have Python 3.6 or later installed. You will also need the following libraries:

- `scikit-learn`
- `scipy`
- `numpy`
- `streamlit` (for creating the web interface)

### Installing Dependencies

1. Clone the repository:

git clone https://github.com/Maharshjiyani/timelytics-delivery-predictor.git
cd timelytics-delivery-predictor
Install the required libraries:

pip install -r requirements.txt
Usage
Run the web app to predict delivery times using Streamlit:

streamlit run app.py
