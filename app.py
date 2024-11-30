from google import drive
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

uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

def preprocess_data(df):
    # Drop unnecessary columns
    columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Convert object columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    df = df.dropna(thresh=int(0.5 * df.shape[0]), axis=1)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle Timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Timestamp_numeric'] = df['Timestamp'].astype(int) / 10**9  # Convert to Unix timestamp
        df['Hour'] = df['Timestamp'].dt.hour
        df = df.drop(columns=['Timestamp'])
    else:
        df['Timestamp_numeric'] = 0
        df['Hour'] = 0

    # Replace infinite values and drop any remaining NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Reindex to ensure same feature order and fill missing columns with zeros
    df = df.reindex(columns=features, fill_value=0)

    # Scale the data
    X_scaled = scaler.transform(df)

    # Since we're not using the selector, return the scaled data
    return X_scaled


if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())

        X_preprocessed = preprocess_data(data)

        # Make predictions
        predictions = model.predict(X_preprocessed)
        data['Prediction'] = predictions

        # Map predictions to labels
        label_mapping = {0: 'Benign', 1: 'Attack'}
        data['Prediction Label'] = data['Prediction'].map(label_mapping)

        st.write("Predictions:")
        st.dataframe(data[['Prediction Label']])

        # Display prediction summary
        st.write("Prediction Summary:")
        prediction_counts = data['Prediction Label'].value_counts()
        st.bar_chart(prediction_counts)

        st.write("Detailed Data with Predictions:")
        st.dataframe(data)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Awaiting CSV file to be uploaded.")

st.header("Chat Bot Interface")


if 'messages' not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", "")

if user_input:
    st.session_state.messages.append({"user": "You", "text": user_input})

    if uploaded_file is not None:
        if 'attack' in user_input.lower():
            attack_count = data['Prediction'].sum()
            response = f"There are {attack_count} potential attacks detected in the uploaded data."
        elif 'benign' in user_input.lower() or 'safe' in user_input.lower():
            benign_count = len(data) - data['Prediction'].sum()
            response = f"There are {benign_count} benign instances in the uploaded data."
        else:
            response = "Please ask about potential attacks or the safety of the network traffic."
    else:
        response = "Please upload data first."

    st.session_state.messages.append({"user": "Bot", "text": response})

for message in st.session_state.messages:
    if message["user"] == "You":
        st.markdown(f"**You:** {message['text']}")
    else:
        st.markdown(f"**Bot:** {message['text']}")


