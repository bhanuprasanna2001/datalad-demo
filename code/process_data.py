# code/process_data.py

import os
import pandas as pd
import datalad.api as dl
from sklearn.model_selection import train_test_split

input_data_path = 'data/raw/diabetes_raw.csv'
train_output_path = 'data/processed/train/diabetes_train.csv'
test_output_path = 'data/processed/test/diabetes_test.csv'

dl.get(input_data_path)

# Load the raw data
raw_data = pd.read_csv(input_data_path)

# Data processing steps on the raw data
# - Handle missing values
# - Encode categorical variables
# - Feature scaling

# Ensure the target variable is integer type
data['Outcome'] = data['Outcome'].astype(int)

# Split the data into features and target
X = raw_data.drop('Outcome', axis=1)  # Adjust 'Outcome' based on your dataset
y = raw_data['Outcome']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

processed_data = raw_data.copy()

# Save the processed data

# Save the processed training data
os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv(train_output_path, index=False)

# Save the processed test data
os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv(test_output_path, index=False)

print(f"Processed training data saved to {train_output_path}")
print(f"Processed test data saved to {test_output_path}")