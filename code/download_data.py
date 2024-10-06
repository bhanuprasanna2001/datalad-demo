# code/download_data.py

import pandas as pd
import urllib.request
import os

# URL to the raw diabetes dataset (e.g., a CSV hosted online)
url = 'https://raw.githubusercontent.com/krishnakishore999/Diabetes_Prediction/refs/heads/main/diabetes.csv'

# Define the path where the data will be saved
raw_data_path = 'data/raw/diabetes_raw.csv'

# Ensure the directory exists
os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

# Download the dataset
urllib.request.urlretrieve(url, raw_data_path)
print(f"Raw data downloaded to {raw_data_path}")
