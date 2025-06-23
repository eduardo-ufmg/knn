import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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


def save_stat_test_result(dataset_name: str, stat_data: dict):
    """
    Saves the statistical comparison results to a JSON file.

    Args:
        dataset_name: The name of the dataset used.
        stat_data: A dictionary containing the statistical test results.
    """
    output_dir = RESULTS_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "statistical_comparison.json"
    with open(output_path, "w") as f:
        json.dump(stat_data, f, indent=4)
    print(f"    -> Saved statistical analysis to '{output_path}'")


def run_single_experiment(dataset_path: Path):
    """
    Runs a comparative experiment for a single dataset using 10-fold cross-validation.

    This function loads a dataset, then for each of the 10 folds, it trains and
    evaluates each model. It captures performance metrics for each fold, calculates
    the average and standard deviation, and saves the aggregated results. Finally,
    it performs a Wilcoxon signed-rank test to compare the models' accuracies
    and saves the statistical result.

    Args:
        dataset_path: The path to the .parquet dataset file.
    """
    dataset_name = dataset_path.stem
    print(f"--- Running experiment on: {dataset_name} ---")

    # 1. Load Data
    data = pd.read_parquet(dataset_path)
    X = data.drop("target", axis=1).values
    y = data["target"].values

    # --- Hyperparameters ---
    K = 15
    H = 1.0
    N_SPLITS = 10

    # --- 2. Setup Cross-Validation ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

    # Lists to store metrics for each fold
    custom_knn_metrics = {"accuracies": [], "train_times": [], "predict_times": []}
    sklearn_knn_metrics = {"accuracies": [], "train_times": [], "predict_times": []}

    # --- 3. Run Cross-Validation Loop ---
    print(f"  Running {N_SPLITS}-Fold Cross-Validation...")
    for fold, (train_index, test_index) in enumerate(skf.split(X, np.asarray(y))):
        print(f"    - Fold {fold + 1}/{N_SPLITS}")
        # Split data for the current fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Pre-process data: Fit on train set, transform both train and test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Evaluate Custom KNNClassifier ---
        custom_knn = KNNClassifier(h=H, k=K, use_support_samples=True)
        start_time = time.perf_counter()
        custom_knn.fit(X_train_scaled, y_train)
        custom_knn_metrics["train_times"].append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        y_pred_custom = custom_knn.predict(X_test_scaled)
        custom_knn_metrics["predict_times"].append(time.perf_counter() - start_time)
        custom_knn_metrics["accuracies"].append(accuracy_score(y_test, y_pred_custom))

        # --- Evaluate Scikit-learn's KNeighborsClassifier ---
        sklearn_knn = SklearnKNN(n_neighbors=K)
        start_time = time.perf_counter()
        sklearn_knn.fit(X_train_scaled, y_train)
        sklearn_knn_metrics["train_times"].append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        y_pred_sklearn = sklearn_knn.predict(X_test_scaled)
        sklearn_knn_metrics["predict_times"].append(time.perf_counter() - start_time)
        sklearn_knn_metrics["accuracies"].append(accuracy_score(y_test, y_pred_sklearn))

    # --- 4. Aggregate and Save Results ---
    print("  Aggregating and saving results...")
    # Custom KNN results
    custom_results = {
        "model_name": "CustomKNNClassifier",
        "dataset": dataset_name,
        "hyperparameters": {"k": K, "h": H, "use_support_samples": True},
        "metrics": {
            "accuracy_mean": np.mean(custom_knn_metrics["accuracies"]),
            "accuracy_std": np.std(custom_knn_metrics["accuracies"]),
            "train_time_mean": np.mean(custom_knn_metrics["train_times"]),
            "train_time_std": np.std(custom_knn_metrics["train_times"]),
            "predict_time_mean": np.mean(custom_knn_metrics["predict_times"]),
            "predict_time_std": np.std(custom_knn_metrics["predict_times"]),
        },
        "fold_metrics": custom_knn_metrics,
    }
    save_result(dataset_name, "custom_knn", custom_results)
    print(
        f"    Custom KNN Mean Accuracy: {custom_results['metrics']['accuracy_mean']:.4f} (+/- {custom_results['metrics']['accuracy_std']:.4f})"
    )

    # Scikit-learn KNN results
    sklearn_results = {
        "model_name": "SklearnKNeighborsClassifier",
        "dataset": dataset_name,
        "hyperparameters": {"n_neighbors": K},
        "metrics": {
            "accuracy_mean": np.mean(sklearn_knn_metrics["accuracies"]),
            "accuracy_std": np.std(sklearn_knn_metrics["accuracies"]),
            "train_time_mean": np.mean(sklearn_knn_metrics["train_times"]),
            "train_time_std": np.std(sklearn_knn_metrics["train_times"]),
            "predict_time_mean": np.mean(sklearn_knn_metrics["predict_times"]),
            "predict_time_std": np.std(sklearn_knn_metrics["predict_times"]),
        },
        "fold_metrics": sklearn_knn_metrics,
    }
    save_result(dataset_name, "sklearn_knn", sklearn_results)
    print(
        f"    Sklearn KNN Mean Accuracy: {sklearn_results['metrics']['accuracy_mean']:.4f} (+/- {sklearn_results['metrics']['accuracy_std']:.4f})"
    )

    # --- 5. Perform Statistical Test for Equivalence ---
    print("  Performing statistical test...")
    alpha = 0.05
    stat, p_value = wilcoxon(
        custom_knn_metrics["accuracies"], sklearn_knn_metrics["accuracies"]
    )
    stat, p_value = cast(float, stat), cast(float, p_value)

    if p_value > alpha:
        conclusion = (
            f"The models are statistically equivalent (p={p_value:.4f} > {alpha})."
        )
    else:
        conclusion = f"There is a significant difference between the models (p={p_value:.4f} <= {alpha})."

    stat_results = {
        "test_name": "Wilcoxon Signed-Rank Test",
        "alpha": alpha,
        "statistic": stat,
        "p_value": p_value,
        "conclusion": conclusion,
    }
    save_stat_test_result(dataset_name, stat_results)
    print(f"    {conclusion}\n")


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
