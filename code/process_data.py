# code/process_data.py

import pandas as pd
import datalad.api as dl

input_data_path = 'data/raw/diabetes_raw.csv'
output_data_path = 'data/processed/diabetes_processed.csv'

dl.get(input_data_path)

# Load the raw data
raw_data = pd.read_csv(input_data_path)

# Data processing steps on the raw data
# - Handle missing values
# - Encode categorical variables
# - Feature scaling

processed_data = raw_data.copy()

# Save the processed data
processed_data.to_csv(output_data_path, index=False)
