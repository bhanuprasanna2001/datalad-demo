# code/process_data.py

import pandas as pd

# Load the raw data
raw_data = pd.read_csv('../data/raw/diabetes_raw.csv')

# Data processing steps on the raw data
# - Handle missing values
# - Encode categorical variables
# - Feature scaling

processed_data = raw_data.copy()

# Save the processed data
processed_data.to_csv('../data/processed/diabetes_processed.csv', index=False)
