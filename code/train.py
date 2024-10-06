# code/train.py

import pandas as pd
import json
import os
import joblib
import sys

def main(experiment_name):
    # Define paths
    train_data_path = 'data/processed/train/diabetes_processed_train.csv'
    params_path = 'params/config.json'
    model_output_path = f'results/{experiment_name}/model.joblib'

    # Load training data
    train_data = pd.read_csv(train_data_path)
    X_train = train_data.drop('Outcome', axis=1)
    y_train = train_data['Outcome']

    # Load parameters
    with open(params_path, 'r') as f:
        config = json.load(f)

    model_name = config.get('model', 'Ridge')
    params = config.get('parameters', {})

    # Initialize the model
    if model_name == 'Ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(**params)
    elif model_name == 'Lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(**params)
    elif model_name == 'ElasticNet':
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <experiment_name>")
        sys.exit(1)
    experiment_name = sys.argv[1]
    main(experiment_name)
