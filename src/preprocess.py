"""
preprocess.py - Loads raw wine data, cleans/splits it, and saves preprocessed data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

RAW_DATA_PATH = "data/raw/wine.csv"
PROCESSED_DIR = "data/processed"
PARAMS_PATH = "params.json"

def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw data
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    # Drop duplicates and handle missing values
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"After cleaning: {df.shape}")

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Load params
    with open(PARAMS_PATH) as f:
        params = json.load(f)

    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    # Save scaler params
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": list(X.columns)
    }
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f)

    # Save processed data
    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.values
    test_df = X_test_scaled.copy()
    test_df["target"] = y_test.values

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess()
