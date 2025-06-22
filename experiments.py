import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.preprocessing import StandardScaler

from knn import KNNClassifier

# Define root directories for datasets and results
DATASETS_DIR = Path("sets/")
RESULTS_DIR = Path("results/")


def save_result(dataset_name: str, model_name: str, result_data: dict):
    """
    Saves a single experiment result to a structured JSON file.

    The results are organized as `results/{dataset_name}/{model_name}.json`

    Args:
        dataset_name: The name of the dataset used.
        model_name: The name of the model evaluated.
        result_data: A dictionary containing the experiment's results.
    """
    # Create the directory for the specific dataset if it doesn't exist
    output_dir = RESULTS_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the output file path
    output_path = output_dir / f"{model_name}.json"

    # Write the results dictionary to the JSON file
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"    -> Saved {model_name} results to '{output_path}'")


def run_single_experiment(dataset_path: Path):
    """
    Runs a comparative experiment for a single dataset.

    This function loads a dataset, trains and evaluates each model,
    and saves their performance results to separate, structured JSON files.

    Args:
        dataset_path: The path to the .parquet dataset file.
    """
    dataset_name = dataset_path.stem
    print(f"--- Running experiment on: {dataset_name} ---")

    # 1. Load and Prepare Data
    data = pd.read_parquet(dataset_path)
    X = data.drop("target", axis=1).values
    y = data["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=np.asarray(y)
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Hyperparameters ---
    K = 15
    H = 1.0

    # --- 2. Evaluate Custom KNNClassifier ---
    print("  Evaluating Custom KNNClassifier...")
    custom_knn = KNNClassifier(h=H, k=K, use_support_samples=True)

    start_time = time.perf_counter()
    custom_knn.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    y_pred_custom = custom_knn.predict(X_test_scaled)
    predict_time = time.perf_counter() - start_time

    # Structure the results for the custom model
    custom_results = {
        "model_name": "CustomKNNClassifier",
        "dataset": dataset_name,
        "hyperparameters": {"k": K, "h": H, "use_support_samples": True},
        "metrics": {
            "accuracy": accuracy_score(y_test, y_pred_custom),
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
        },
        "metadata": {
            "original_train_samples": X_train_scaled.shape[0],
            "reference_samples_used": (
                custom_knn.X_ref_.shape[0]
                if hasattr(custom_knn, "X_ref_")
                else X_train_scaled.shape[0]
            ),
        },
    }
    save_result(dataset_name, "custom_knn", custom_results)
    print(f"    Custom KNN Accuracy: {custom_results['metrics']['accuracy']:.4f}")

    # --- 3. Evaluate Scikit-learn's KNeighborsClassifier ---
    print("  Evaluating Sklearn KNeighborsClassifier...")
    sklearn_knn = SklearnKNN(n_neighbors=K)

    start_time = time.perf_counter()
    sklearn_knn.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    y_pred_sklearn = sklearn_knn.predict(X_test_scaled)
    predict_time = time.perf_counter() - start_time

    # Structure the results for the scikit-learn model
    sklearn_results = {
        "model_name": "SklearnKNeighborsClassifier",
        "dataset": dataset_name,
        "hyperparameters": {"n_neighbors": K},
        "metrics": {
            "accuracy": accuracy_score(y_test, y_pred_sklearn),
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
        },
        "metadata": {"original_train_samples": X_train_scaled.shape[0]},
    }
    save_result(dataset_name, "sklearn_knn", sklearn_results)
    print(f"    Sklearn KNN Accuracy: {sklearn_results['metrics']['accuracy']:.4f}\n")


def main():
    """
    Main function to find datasets and orchestrate the experiments.
    """
    # Verify dataset directory exists
    if not DATASETS_DIR.exists() or not DATASETS_DIR.is_dir():
        print(f"Error: Dataset directory not found at '{DATASETS_DIR}'")
        print("Please run 'python store_datasets/store_sets.py sets/' first.")
        return

    # Find all .parquet dataset files
    dataset_paths = sorted(list(DATASETS_DIR.glob("*.parquet")))
    if not dataset_paths:
        print(f"Error: No .parquet datasets found in '{DATASETS_DIR}'")
        return

    # Create the main results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print(" " * 25 + "--- STARTING EXPERIMENTS ---")
    print("=" * 80)

    for path in dataset_paths:
        try:
            run_single_experiment(path)
        except Exception as e:
            print(f"\n[ERROR] Failed to run experiment on '{path.stem}'.")
            print(f"  Reason: {e}\n")

    print("=" * 80)
    print("--- ALL EXPERIMENTS COMPLETE ---")
    print(f"All results have been saved in the '{RESULTS_DIR}' directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
