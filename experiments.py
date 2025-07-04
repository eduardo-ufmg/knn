import json
import time
from pathlib import Path
from typing import Any, cast
from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from CorrelationFilter.CorrelationFilter import CorrelationFilter
from parameterless_knn import ParameterlessKNNClassifier
from sklearn_knn_parameterless_wrapper import SklearnKNNParameterlessWrapper

# Define root directories for datasets and results
DATASETS_DIR = Path("sets/")
RESULTS_DIR = Path("results/")


def save_result(dataset_name: str, model_name: str, result_data: dict):
    """
    Saves a single experiment result to a structured JSON file.

    The results are organized as `results/{dataset_name}/{model_name}.json`

    Parameters:
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


def save_all_stat_tests(dataset_name: str, all_stats_data: list[dict]):
    """
    Saves all statistical comparison results to a single JSON file.

    Parameters:
        dataset_name: The name of the dataset used.
        all_stats_data: A list of dictionaries, each containing the results of one statistical test.
    """
    output_dir = RESULTS_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "statistical_comparisons.json"
    with open(output_path, "w") as f:
        json.dump(all_stats_data, f, indent=4)
    print(f"    -> Saved all statistical analyses to '{output_path}'")


def run_single_experiment(dataset_path: Path):
    """
    Runs a comparative experiment for a single dataset using 10-fold cross-validation.

    This function evaluates multiple parameter-less KNN models. For each model,
    it performs 10-fold cross-validation to measure performance. After evaluating
    all models, it performs Wilcoxon signed-rank tests to compare each custom
    optimization metric against the scikit-learn based wrapper.

    Parameters:
        dataset_path: The path to the .parquet dataset file.
    """
    dataset_name = dataset_path.stem
    print(f"--- Running experiment on: {dataset_name} ---")

    # 1. Load Data
    data = pd.read_parquet(dataset_path)
    X = data.drop("target", axis=1).values
    y = data["target"].values

    # --- Experiment Configuration ---
    N_SPLITS = 10
    N_OPTIMIZER_CALLS = 25  # Number of calls for Bayesian Optimization

    # --- 2. Define Models for Evaluation ---
    models_to_evaluate: list[tuple[str, Any]] = []

    # Define all options for metrics and support sample methods
    metrics = [
        "dissimilarity",
        "silhouette",
        "spread",
        "convex_hull_inter",
        "convex_hull_intra",
        "opposite_hyperplane",
    ]
    support_samples_methods = ["hnbf", "margin_clustering", "gabriel_graph"]

    # Generate a model for each combination of metric and support sample method
    for metric in metrics:
        for ss_method in support_samples_methods:
            model_name = f"parameterless_knn_{metric}_{ss_method}"
            model_instance = ParameterlessKNNClassifier(
                metric=metric,
                support_samples_method=ss_method,
                n_optimizer_calls=N_OPTIMIZER_CALLS,
            )
            models_to_evaluate.append((model_name, model_instance))

    # Add the scikit-learn wrapper as a baseline for comparison
    models_to_evaluate.append(
        (
            "sklearn_knn_wrapper",
            SklearnKNNParameterlessWrapper(n_optimizer_calls=N_OPTIMIZER_CALLS),
        )
    )

    # --- 3. Run Cross-Validation for Each Model ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    all_fold_accuracies = {}

    print(f"  Running {N_SPLITS}-Fold Cross-Validation for each model...")
    for model_name, model in models_to_evaluate:
        print(f"\n  Evaluating model: {model_name}")
        fold_metrics = {"accuracies": [], "train_times": [], "predict_times": []}

        for fold, (train_index, test_index) in enumerate(skf.split(X, np.asarray(y))):
            print(f"    - Fold {fold + 1}/{N_SPLITS}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Pre-process data: Fit on train set, transform both using a pipeline
            preprocess_pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("variance_threshold", VarianceThreshold(threshold=0.1)),
                    ("correlation_filter", CorrelationFilter(threshold=0.9)),
                    ("pca", PCA(n_components=0.9)),
                ]
            )
            X_train_scaled = preprocess_pipeline.fit_transform(X_train)
            X_test_scaled = preprocess_pipeline.transform(X_test)

            try:
                # --- Train ---
                start_time = time.perf_counter()
                model.fit(X_train_scaled, y_train)
                fold_metrics["train_times"].append(time.perf_counter() - start_time)

                # --- Predict ---
                start_time = time.perf_counter()
                y_pred = model.predict(X_test_scaled)
                fold_metrics["predict_times"].append(time.perf_counter() - start_time)
                fold_metrics["accuracies"].append(accuracy_score(y_test, y_pred))

            except Exception as e:
                warn(f"  [Warning] Model '{model_name}' failed on fold {fold + 1}: {e}")
                fold_metrics["train_times"].append(0.0)
                fold_metrics["predict_times"].append(0.0)
                fold_metrics["accuracies"].append(0.0)
                continue

        # --- Aggregate and Save Results for the current model ---
        model_results = {
            "model_name": model_name,
            "dataset": dataset_name,
            "hyperparameters": model.get_params(),
            "metrics": {
                "accuracy_mean": np.mean(fold_metrics["accuracies"]),
                "accuracy_std": np.std(fold_metrics["accuracies"]),
                "train_time_mean": np.mean(fold_metrics["train_times"]),
                "train_time_std": np.std(fold_metrics["train_times"]),
                "predict_time_mean": np.mean(fold_metrics["predict_times"]),
                "predict_time_std": np.std(fold_metrics["predict_times"]),
            },
            "fold_metrics": fold_metrics,
        }
        save_result(dataset_name, model_name, model_results)
        print(
            f"    -> Mean Accuracy: {model_results['metrics']['accuracy_mean']:.4f} "
            f"(+/- {model_results['metrics']['accuracy_std']:.4f})"
        )

        # Store accuracies for the final statistical comparison
        all_fold_accuracies[model_name] = fold_metrics["accuracies"]

    # --- 4. Perform and Save Statistical Comparisons ---
    print("\n  Performing statistical tests...")
    statistical_results = []
    reference_model_name = "sklearn_knn_wrapper"
    reference_accuracies = all_fold_accuracies.get(reference_model_name)

    if reference_accuracies is None:
        print(
            f"  [Warning] Reference model '{reference_model_name}' not found. "
            "Skipping statistical tests."
        )
        return

    # Compare each custom model to the reference scikit-learn wrapper
    for model_name, accuracies in all_fold_accuracies.items():
        if model_name == reference_model_name:
            continue

        print(f"    - Comparing '{model_name}' vs '{reference_model_name}'")
        alpha = 0.05
        stat, p_value, conclusion = None, None, ""

        try:
            # Perform the Wilcoxon signed-rank test
            stat, p_value = wilcoxon(
                accuracies, reference_accuracies, zero_method="zsplit"
            )
            stat, p_value = cast(float, stat), cast(float, p_value)

            # Interpret the p-value
            if p_value > alpha:
                conclusion = f"The models are statistically equivalent (p={p_value:.4f} > {alpha})."
            else:
                mean_diff = np.mean(accuracies) - np.mean(reference_accuracies)
                if mean_diff > 0:
                    conclusion = f"'{model_name}' is significantly better than '{reference_model_name}' (p={p_value:.4f} <= {alpha})."
                else:
                    conclusion = f"'{reference_model_name}' is significantly better than '{model_name}' (p={p_value:.4f} <= {alpha})."

        except ValueError:
            # This can occur if all accuracy differences are zero
            stat, p_value = 0.0, 1.0
            conclusion = "Could not perform Wilcoxon test as all differences are zero. Models are equivalent."

        stat_result = {
            "test_name": "Wilcoxon Signed-Rank Test",
            "comparison": f"{model_name}_vs_{reference_model_name}",
            "model_1": model_name,
            "model_2": reference_model_name,
            "alpha": alpha,
            "statistic": stat,
            "p_value": p_value,
            "conclusion": conclusion,
        }
        statistical_results.append(stat_result)
        print(f"      {conclusion}")

    # Save all statistical results to a single file for the dataset
    if statistical_results:
        save_all_stat_tests(dataset_name, statistical_results)


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
