import gdown
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import os

st.title("Network Traffic Classification App")
st.write("""
This application allows you to upload network traffic data, run a pre-trained model, and receive threat predictions.
""")

# URLs to your model artifacts
model_url = 'https://raw.githubusercontent.com/AnthonyBurton-Cordova/SLADA_Project/main/neural_network_model.joblib'
scaler_url = 'https://raw.githubusercontent.com/AnthonyBurton-Cordova/SLADA_Project/main/scaler.joblib'
features_url = 'https://raw.githubusercontent.com/AnthonyBurton-Cordova/SLADA_Project/main/features_list.joblib'

# Selector file from Google Drive
selector_file_id = '1409kJ5YR9rCcui9fo-LoiA_xoEvVj7Kq'
selector_destination = 'selector.joblib'
selector_url = f"https://drive.google.com/uc?id={selector_file_id}"

if os.path.exists(selector_destination):
    st.write(f"'{selector_destination}' already exists, skipping download.")
else:
    try:
        gdown.download(selector_url, selector_destination, quiet=False)
        st.success(f"Selector downloaded successfully as '{selector_destination}'.")
    except Exception as e:
        st.error(f"An error occurred while downloading the selector: {e}")

# Function to load joblib files from URL
def load_joblib_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading joblib file from {url}: {e}")
        return None

# Load the artifacts from the URLs
model = load_joblib_from_url(model_url)
scaler = load_joblib_from_url(scaler_url)
features = load_joblib_from_url(features_url)
selector = load_joblib_from_url(features_url)

uploaded_file = st.file_uploader("Please upload a CSV file of network traffic data for analysis", type="csv")

def preprocess_data(df):
    try:
        # Drop unnecessary columns
        columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Convert object columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'Label':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        df = df.dropna(thresh=int(0.5 * df.shape[0]), axis=1)
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df.loc[:, col] = df[col].fillna(df[col].median())

        # Drop duplicates
        df = df.drop_duplicates()

        # Handle Timestamp
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df.loc[:, 'Timestamp_numeric'] = df['Timestamp'].astype(int) / 10**9
                df.loc[:, 'Hour'] = df['Timestamp'].dt.hour
            except Exception as e:
                st.warning(f"Timestamp conversion failed: {e}")
            finally:
                df = df.drop(columns=['Timestamp'], errors='ignore')
        else:
            df['Timestamp_numeric'] = 0
            df['Hour'] = 0

        # Replace infinite values and drop any remaining NaNs
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Ensure same feature order and fill missing columns with zeros
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.warning(f"Missing features: {missing_features}. These will be filled with zeros.")
        df = df.reindex(columns=features, fill_value=0)

        # Scale the data
        X_scaled = scaler.transform(df)

        # Feature selection
        X_selected = selector.transform(X_scaled)

        return X_selected
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head(10))

        X_preprocessed = preprocess_data(data)

        if X_preprocessed is not None:
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

            # Provide download link for predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
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

    if uploaded_file is not None and 'Prediction' in data:
        if 'attack' in user_input.lower():
            attack_count = data['Prediction'].sum()
            response = f"There are {attack_count} potential attacks detected in the uploaded data."
        elif 'benign' in user_input.lower() or 'safe' in user_input.lower():
            benign_count = len(data) - data['Prediction'].sum()
            response = f"There are {benign_count} benign instances in the uploaded data."
        elif 'summary' in user_input.lower():
            attack_count = data['Prediction'].sum()
            benign_count = len(data) - attack_count
            response = f"Summary: {attack_count} potential attacks and {benign_count} benign instances."
        elif 'help' in user_input.lower():
            response = "You can ask about attacks, benign instances, or request a summary of the predictions."
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
