# code/evaluate.py

import os
import sys
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

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
plot_output_path = f'results/{experiment_name}/roc_curve.png'

# Load test data
test_data = pd.read_csv(test_data_path)
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']

# Load the trained model
model = joblib.load(model_path)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions, output_dict=True)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Save metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': conf_matrix.tolist(),
    'classification_report': class_report
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

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(plot_output_path)
plt.close()

print(f"Evaluation metrics saved to {metrics_output_path}")
print(f"Predictions saved to {predictions_output_path}")
print(f"ROC curve plot saved to {plot_output_path}")