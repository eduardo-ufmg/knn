import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.preprocessing import StandardScaler

# --- Important ---
# Ensure that this script is run from the root of your project directory
# so that it can correctly import the 'knn' module.
from knn import KNNClassifier

# Define the directory where datasets are stored
DATASETS_DIR = Path("sets/")


def run_single_experiment(dataset_path: Path):
    """
    Runs a comparative experiment for a single dataset.

    This function loads a dataset, splits it, and then trains and evaluates
    both the custom KNNClassifier and scikit-learn's KNeighborsClassifier,
    recording performance metrics for each.

    Args:
        dataset_path: The path to the .parquet dataset file.

    Returns:
        A dictionary containing the performance results for the dataset.
    """
    print(f"--- Running experiment on: {dataset_path.stem} ---")

    # 1. Load and Prepare Data
    data = pd.read_parquet(dataset_path)
    X = data.drop("target", axis=1).values
    y = data["target"].values

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    # Scale data for fair comparison in distance-based algorithms
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Hyperparameters for a Fair Comparison ---
    K = 15  # Number of neighbors for both classifiers
    H = 1.0  # Bandwidth for our custom classifier's RBF kernel

    results = {"dataset": dataset_path.stem}

    # --- 4. Evaluate Custom KNNClassifier ---
    print("Evaluating Custom KNNClassifier...")
    custom_knn = KNNClassifier(h=H, k=K, use_support_samples=True)

    # Train
    start_time = time.perf_counter()
    custom_knn.fit(X_train_scaled, y_train)
    end_time = time.perf_counter()
    results["custom_train_time"] = end_time - start_time

    # Predict
    start_time = time.perf_counter()
    y_pred_custom = custom_knn.predict(X_test_scaled)
    end_time = time.perf_counter()
    results["custom_predict_time"] = end_time - start_time

    # Record metrics
    results["custom_accuracy"] = accuracy_score(y_test, y_pred_custom)
    # Report the number of reference samples used after reduction
    if hasattr(custom_knn, "X_ref_"):
        results["custom_ref_samples"] = custom_knn.X_ref_.shape[0]
    else:
        results["custom_ref_samples"] = X_train_scaled.shape[0]
    results["original_train_samples"] = X_train_scaled.shape[0]

    # --- 5. Evaluate Scikit-learn's KNeighborsClassifier ---
    print("Evaluating Sklearn KNeighborsClassifier...")
    # Using 'uniform' weights for the most direct comparison
    sklearn_knn = SklearnKNN(n_neighbors=K, weights="uniform", algorithm="auto")

    # Train
    start_time = time.perf_counter()
    sklearn_knn.fit(X_train_scaled, y_train)
    end_time = time.perf_counter()
    results["sklearn_train_time"] = end_time - start_time

    # Predict
    start_time = time.perf_counter()
    y_pred_sklearn = sklearn_knn.predict(X_test_scaled)
    end_time = time.perf_counter()
    results["sklearn_predict_time"] = end_time - start_time

    # Record metrics
    results["sklearn_accuracy"] = accuracy_score(y_test, y_pred_sklearn)

    print(f"Custom KNN Accuracy: {results['custom_accuracy']:.4f}")
    print(f"Sklearn KNN Accuracy: {results['sklearn_accuracy']:.4f}\n")

    return results


def display_results(all_results: list):
    """Formats and prints the final results table."""
    if not all_results:
        print("No results to display.")
        return

    results_df = pd.DataFrame(all_results).set_index("dataset")

    # Reorder columns for better readability and comparison
    cols_order = [
        "original_train_samples",
        "custom_ref_samples",
        "custom_accuracy",
        "sklearn_accuracy",
        "custom_train_time",
        "sklearn_train_time",
        "custom_predict_time",
        "sklearn_predict_time",
    ]
    results_df = results_df[cols_order]

    # Add insightful comparison columns
    results_df["accuracy_diff"] = (
        results_df["custom_accuracy"] - results_df["sklearn_accuracy"]
    )
    results_df["train_time_ratio"] = (
        results_df["custom_train_time"] / results_df["sklearn_train_time"]
    )
    results_df["predict_time_ratio"] = (
        results_df["custom_predict_time"] / results_df["sklearn_predict_time"]
    )

    print("\n" + "=" * 80)
    print(" " * 25 + "--- EXPERIMENT RESULTS ---")
    print("=" * 80)
    # Set display options for clean output
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        120,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(results_df)

    print("\n" + "=" * 80)
    print("Notes on Ratio Columns:")
    print("- A value > 1.0 means the custom KNN was slower for that metric.")
    print("- A value < 1.0 means the custom KNN was faster.")
    print("- `custom_ref_samples` shows the effect of the support sample reduction.")
    print("=" * 80)


def main():
    """
    Main function to find datasets and orchestrate the experiments.
    """
    # Verify that the dataset directory exists
    if not DATASETS_DIR.exists() or not DATASETS_DIR.is_dir():
        print(f"Error: Dataset directory not found at '{DATASETS_DIR}'")
        print("Please run 'python store_datasets/store_sets.py sets/' first.")
        return

    # Find all .parquet dataset files
    dataset_paths = sorted(list(DATASETS_DIR.glob("*.parquet")))
    if not dataset_paths:
        print(f"Error: No .parquet datasets found in '{DATASETS_DIR}'")
        return

    all_results = []
    for path in dataset_paths:
        try:
            result = run_single_experiment(path)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to run experiment on '{path.stem}'.")
            print(f"  Reason: {e}\n")

    # Display the final summary table
    display_results(all_results)


if __name__ == "__main__":
    main()
