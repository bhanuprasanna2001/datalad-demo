# code/download_data.py

import pandas as pd
import urllib.request

# URL to the raw diabetes dataset (e.g., a CSV hosted online)
url = 'https://raw.githubusercontent.com/krishnakishore999/Diabetes_Prediction/refs/heads/main/diabetes.csv'

# Download the dataset
raw_data_path = '../data/raw/diabetes_raw.csv'
urllib.request.urlretrieve(url, raw_data_path)