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
