import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import os

st.title("Network Traffic Classification App")
st.write("""
This application allows you to upload network traffic data, run a pre-trained model, and receive predictions.
""")

# URLs to your model artifacts (excluding the selector)
model_url = 'https://raw.githubusercontent.com/AnthonyBurton-Cordova/SLADA_Project/main/neural_network_model.joblib'
scaler_url = 'https://raw.githubusercontent.com/AnthonyBurton-Cordova/SLADA_Project/main/scaler.joblib'
features_url = 'https://raw.githubusercontent.com/AnthonyBurton-Cordova/SLADA_Project/main/features_list.joblib'

# Function to load joblib files from URL
def load_joblib_from_url(url):
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

# Load the artifacts from the URLs
model = load_joblib_from_url(model_url)
scaler = load_joblib_from_url(scaler_url)
features = load_joblib_from_url(features_url)
