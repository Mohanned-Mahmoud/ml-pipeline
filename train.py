"""
train.py - Loads train.csv, trains a Random Forest model, logs to MLflow, saves the model.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import pickle
import os
import json

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"
PARAMS_PATH = "params.json"


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    print("Loading preprocessed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # Load params
    with open(PARAMS_PATH) as f:
        params = json.load(f)

    rf_params = params["train"]["random_forest"]

    # Configure MLflow
    mlflow.set_experiment("wine-classification")

    with mlflow.start_run(run_name="random_forest_baseline"):
        # Log params
        mlflow.log_params(rf_params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train model
        print(f"Training Random Forest with params: {rf_params}")
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log feature importances
        feature_importance = dict(zip(
            X_train.columns,
            model.feature_importances_.tolist()
        ))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in top_features:
            mlflow.log_metric(f"importance_{feat[:15]}", imp)

        # Save model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Also save locally as pickle
        model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)

        print(f"\nModel saved to {model_path}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    print("Training complete!")


if __name__ == "__main__":
    train()
