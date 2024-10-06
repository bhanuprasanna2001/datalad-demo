# code/evaluate.py

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Check for the correct number of arguments
if len(sys.argv) != 2:
    print("Usage: python evaluate.py <experiment_name>")
    sys.exit(1)

# Get the experiment name from the command-line arguments
experiment_name = sys.argv[1]

# Define paths
test_data_path = 'data/processed/test/diabetes_test.csv'
model_path = f'results/{experiment_name}/model.joblib'
metrics_output_path = f'results/{experiment_name}/metrics.json'
predictions_output_path = f'results/{experiment_name}/predictions.csv'

# Load test data
test_data = pd.read_csv(test_data_path)
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']

# Load the trained model
model = joblib.load(model_path)

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save metrics
metrics = {
    'mse': mse,
    'r2_score': r2
}
os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
with open(metrics_output_path, 'w') as f:
    json.dump(metrics, f, indent=4)

# Save predictions
pred_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})
pred_df.to_csv(predictions_output_path, index=False)

print(f"Evaluation metrics saved to {metrics_output_path}")
print(f"Predictions saved to {predictions_output_path}")