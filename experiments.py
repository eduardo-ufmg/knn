import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    output_dir = RESULTS_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.json"
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


def evaluate_model(
    model_tuple: tuple[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_splits: int,
):
    """
    Evaluates a single model using cross-validation.

    This function is designed to be a self-contained task that can be run in a separate process.

    Parameters:
        model_tuple: A tuple containing the model name and the model instance.
        X: Feature data.
        y: Target labels.
        dataset_name: The name of the dataset being processed.
        n_splits: The number of cross-validation folds.

    Returns:
        A tuple containing the model name and a dictionary of its results.
    """
    model_name, model = model_tuple
    print(f"\n  Evaluating model: {model_name} on dataset: {dataset_name}")
    fold_metrics = {"accuracies": [], "train_times": [], "predict_times": []}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    for fold, (train_index, test_index) in enumerate(skf.split(X, np.asarray(y))):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
            start_time = time.perf_counter()
            model.fit(X_train_scaled, y_train)
            fold_metrics["train_times"].append(time.perf_counter() - start_time)

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
        f"    -> {model_name} on {dataset_name} Mean Accuracy: {model_results['metrics']['accuracy_mean']:.4f} "
        f"(+/- {model_results['metrics']['accuracy_std']:.4f})"
    )
    return model_name, model_results


def run_single_experiment(dataset_path: Path, parallel_models: bool = False):
    """
    Runs a comparative experiment for a single dataset.

    This function can evaluate models sequentially or in parallel, controlled by the
    `parallel_models` flag.

    Parameters:
        dataset_path: The path to the .parquet dataset file.
        parallel_models: If True, evaluates models in parallel. Defaults to False.
    """
    dataset_name = dataset_path.stem
    print(f"--- Running experiment on: {dataset_name} ---")

    data = pd.read_parquet(dataset_path)
    X = data.drop("target", axis=1).values
    y = data["target"].to_numpy()

    N_SPLITS = 10
    N_OPTIMIZER_CALLS = 25

    models_to_evaluate: list[tuple[str, Any]] = []
    metrics = [
        "dissimilarity",
        "silhouette",
        "spread",
        "convex_hull_inter",
        "convex_hull_intra",
        "opposite_hyperplane",
    ]
    support_samples_methods = ["hnbf", "margin_clustering", "gabriel_graph"]

    for metric in metrics:
        for ss_method in support_samples_methods:
            model_name = f"parameterless_knn_{metric}_{ss_method}"
            models_to_evaluate.append(
                (
                    model_name,
                    ParameterlessKNNClassifier(
                        metric=metric,
                        support_samples_method=ss_method,
                        n_optimizer_calls=N_OPTIMIZER_CALLS,
                    ),
                )
            )

    models_to_evaluate.append(
        (
            "sklearn_knn_wrapper",
            SklearnKNNParameterlessWrapper(n_optimizer_calls=N_OPTIMIZER_CALLS),
        )
    )

    all_fold_accuracies = {}

    if parallel_models:
        print(
            f"  Running {N_SPLITS}-Fold Cross-Validation in PARALLEL for each model..."
        )
        # Use a fraction of cores to avoid overwhelming the system
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(
                    evaluate_model, model_tuple, X, y, dataset_name, N_SPLITS
                ): model_tuple[0]
                for model_tuple in models_to_evaluate
            }

            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    _, model_results = future.result()
                    all_fold_accuracies[model_name] = model_results["fold_metrics"][
                        "accuracies"
                    ]
                except Exception as exc:
                    print(
                        f"  [ERROR] Model '{model_name}' on dataset '{dataset_name}' generated an exception: {exc}"
                    )

    else:
        print(
            f"  Running {N_SPLITS}-Fold Cross-Validation SEQUENTIALLY for each model..."
        )
        for model_tuple in models_to_evaluate:
            try:
                model_name, model_results = evaluate_model(
                    model_tuple, X, y, dataset_name, N_SPLITS
                )
                all_fold_accuracies[model_name] = model_results["fold_metrics"][
                    "accuracies"
                ]
            except Exception as exc:
                print(
                    f"  [ERROR] Evaluation failed for model '{model_tuple[0]}': {exc}"
                )

    print("\n  Performing statistical tests...")
    statistical_results = []
    reference_model_name = "sklearn_knn_wrapper"
    reference_accuracies = all_fold_accuracies.get(reference_model_name)

    if reference_accuracies is None:
        print(
            f"  [Warning] Reference model '{reference_model_name}' not found. Skipping statistical tests."
        )
        return

    for model_name, accuracies in all_fold_accuracies.items():
        if model_name == reference_model_name:
            continue

        print(f"    - Comparing '{model_name}' vs '{reference_model_name}'")
        alpha = 0.05
        stat, p_value, conclusion = None, None, ""

        try:
            stat, p_value = wilcoxon(
                accuracies, reference_accuracies, zero_method="zsplit"
            )
            stat, p_value = cast(float, stat), cast(float, p_value)

            if p_value > alpha:
                conclusion = f"The models are statistically equivalent (p={p_value:.4f} > {alpha})."
            else:
                mean_diff = np.mean(accuracies) - np.mean(reference_accuracies)
                conclusion = (
                    f"'{model_name}' is significantly better than '{reference_model_name}' (p={p_value:.4f} <= {alpha})."
                    if mean_diff > 0
                    else f"'{reference_model_name}' is significantly better than '{model_name}' (p={p_value:.4f} <= {alpha})."
                )
        except ValueError:
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

    if statistical_results:
        save_all_stat_tests(dataset_name, statistical_results)


def main():
    """
    Main function to find datasets and orchestrate the experiments with parallel execution.
    """
    if not DATASETS_DIR.exists() or not DATASETS_DIR.is_dir():
        print(f"Error: Dataset directory not found at '{DATASETS_DIR}'")
        print("Please run 'python store_datasets/store_sets.py sets/' first.")
        return

    dataset_paths = sorted(list(DATASETS_DIR.glob("*.parquet")))
    if not dataset_paths:
        print(f"Error: No .parquet datasets found in '{DATASETS_DIR}'")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 80)
    print(" " * 25 + "--- STARTING EXPERIMENTS ---")
    print("=" * 80)

    # Dynamically choose parallelization strategy
    if len(dataset_paths) > 1:
        # Strategy 1: Parallelize by dataset
        print(
            f"Found {len(dataset_paths)} datasets. Running experiments in parallel for each dataset."
        )
        max_workers = min(len(dataset_paths), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(run_single_experiment, path): path
                for path in dataset_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    future.result()  # Retrieve result to raise any exceptions
                except Exception as exc:
                    print(
                        f"\n[ERROR] Dataset '{path.stem}' failed with an exception: {exc}"
                    )
    elif len(dataset_paths) == 1:
        # Strategy 2: Parallelize by model within the single dataset
        print("Found 1 dataset. Running experiments in parallel for each model.")
        run_single_experiment(dataset_paths[0], parallel_models=True)

    print("=" * 80)
    print("--- ALL EXPERIMENTS COMPLETE ---")
    print(f"All results have been saved in the '{RESULTS_DIR}' directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
