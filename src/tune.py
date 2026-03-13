"""
tune.py - Uses MLflow nested runs to tune hyperparameters for a Gradient Boosting model.
Tests at least two hyperparameters using grid search with nested MLflow runs.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import pickle
import os
import json
import itertools

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"
PARAMS_PATH = "params.json"


def tune():
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

    tune_params = params["tune"]
    n_estimators_list = tune_params["n_estimators"]
    learning_rate_list = tune_params["learning_rate"]
    max_depth_list = tune_params["max_depth"]

    # Configure MLflow
    mlflow.set_experiment("wine-classification")

    best_acc = 0
    best_model = None
    best_run_id = None
    results = []

    # Parent run for the full tuning job
    with mlflow.start_run(run_name="gradient_boosting_tuning") as parent_run:
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("tuning_strategy", "grid_search")
        mlflow.log_param("n_estimators_options", str(n_estimators_list))
        mlflow.log_param("learning_rate_options", str(learning_rate_list))
        mlflow.log_param("max_depth_options", str(max_depth_list))

        combos = list(itertools.product(n_estimators_list, learning_rate_list, max_depth_list))
        print(f"\nTuning {len(combos)} hyperparameter combinations with nested MLflow runs...\n")

        for n_est, lr, depth in combos:
            run_name = f"GB_n{n_est}_lr{lr}_d{depth}"
            print(f"  Running: {run_name}")

            # Nested run for each hyperparameter combination
            with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                # Log hyperparameters
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("max_depth", depth)
                mlflow.log_param("random_state", 42)

                # Train model
                model = GradientBoostingClassifier(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=depth,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # Test set evaluation
                y_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average="weighted")

                # Log metrics
                mlflow.log_metric("cv_accuracy_mean", cv_mean)
                mlflow.log_metric("cv_accuracy_std", cv_std)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("test_f1_score", test_f1)

                # Log model artifact
                mlflow.sklearn.log_model(model, "gradient_boosting_model")

                print(f"    CV Acc: {cv_mean:.4f} ± {cv_std:.4f} | Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

                results.append({
                    "n_estimators": n_est,
                    "learning_rate": lr,
                    "max_depth": depth,
                    "cv_accuracy": cv_mean,
                    "test_accuracy": test_acc,
                    "test_f1": test_f1,
                    "run_id": child_run.info.run_id
                })

                # Track best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = model
                    best_run_id = child_run.info.run_id

        # Log best results in parent run
        best_result = max(results, key=lambda x: x["test_accuracy"])
        mlflow.log_metric("best_test_accuracy", best_result["test_accuracy"])
        mlflow.log_metric("best_cv_accuracy", best_result["cv_accuracy"])
        mlflow.log_param("best_n_estimators", best_result["n_estimators"])
        mlflow.log_param("best_learning_rate", best_result["learning_rate"])
        mlflow.log_param("best_max_depth", best_result["max_depth"])
        mlflow.log_param("best_child_run_id", best_result["run_id"])
        mlflow.log_param("total_combinations_tested", len(combos))

        # Save best model
        model_path = os.path.join(MODEL_DIR, "gradient_boosting_best.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(model_path)

        # Save results summary
        results_df = pd.DataFrame(results)
        results_path = os.path.join(MODEL_DIR, "tuning_results.csv")
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        print(f"\n{'='*60}")
        print(f"TUNING COMPLETE - Best Configuration:")
        print(f"  n_estimators:  {best_result['n_estimators']}")
        print(f"  learning_rate: {best_result['learning_rate']}")
        print(f"  max_depth:     {best_result['max_depth']}")
        print(f"  Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"  CV Accuracy:   {best_result['cv_accuracy']:.4f}")
        print(f"  Best Run ID:   {best_result['run_id']}")
        print(f"  Parent Run ID: {parent_run.info.run_id}")
        print(f"{'='*60}")

    print("\nHyperparameter tuning complete!")


if __name__ == "__main__":
    tune()
