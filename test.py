import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from CorrelationFilter.CorrelationFilter import CorrelationFilter
from parameterless_knn import ParameterlessKNNClassifier
from sklearn_knn_parameterless_wrapper import SklearnKNNParameterlessWrapper


def run_single_experiment(
    classifier, X_train, X_test, y_train, y_test, n_classes, experiment_name, cm=False
):
    """
    Trains and evaluates a single classifier, returning its performance metrics.

    Parameters:
        classifier: An unfitted classifier instance.
        X_train, X_test, y_train, y_test: The training and testing data.
        n_classes (int): The number of classes in the dataset.
        experiment_name (str): The name of the experiment for titles and logs.
        cm (bool): Whether to display the confusion matrix. Defaults to False.

    Returns:
        dict: A dictionary containing the performance metrics for the model.
    """
    print("\n" + "=" * 60)
    print(f"Running Experiment: {experiment_name}")
    print("=" * 60)

    # --- 1. Data Preprocessing ---
    preprocess_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("variance_threshold", VarianceThreshold(0.1)),
            ("corelation_filter", CorrelationFilter(0.9)),
            ("pca", PCA(0.9)),
        ]
    )
    X_train = preprocess_pipeline.fit_transform(X_train)
    X_test = preprocess_pipeline.transform(X_test)

    # --- 2. Model Training ---
    print(f"Training classifier: {classifier.__class__.__name__}")
    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.4f} seconds.")

    # --- 3. Prediction ---
    print("\nMaking predictions on the test set...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time
    print(f"Prediction completed in {prediction_time:.4f} seconds.")

    # --- 4. Evaluation & Data Collection ---
    print("\n--- Classifier Performance ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Initialize results dictionary
    results = {
        "training_time": training_time,
        "prediction_time": prediction_time,
        "overall_accuracy": accuracy,
        "optimal_parameters": None,
        "reference_samples_used": None,
    }

    # Store optimal parameters
    if isinstance(classifier, SklearnKNNParameterlessWrapper) and hasattr(
        classifier, "best_params_"
    ):
        results["optimal_parameters"] = classifier.best_params_
        print(f"Best Scikit-learn KNN params found: {classifier.best_params_}")
    elif (
        isinstance(classifier, ParameterlessKNNClassifier)
        and hasattr(classifier, "h_")
        and hasattr(classifier, "k_")
    ):
        results["optimal_parameters"] = {
            "h": float(classifier.h_),
            "k": int(classifier.k_),
        }
        print(f"Best custom KNN params found: h={classifier.h_:.4f}, k={classifier.k_}")

    # Store reference samples count
    if isinstance(classifier, ParameterlessKNNClassifier) and hasattr(
        classifier, "X_ref_"
    ):
        num_ref_samples = classifier.X_ref_.shape[0]
        results["reference_samples_used"] = int(num_ref_samples)
        print(
            f"Classifier is using {num_ref_samples} reference samples (out of "
            f"{len(X_train)} original training samples)."
        )

    # --- 5. Visualization (Optional) ---
    if cm:
        print("Generating confusion matrix...")
        # ... (visualization code remains the same)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(n_classes)],
            yticklabels=[f"Class {i}" for i in range(n_classes)],
        )
        plt.title(f"Confusion Matrix for {experiment_name}")
        plt.ylabel("Actual Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.show()

    return results


def run_all_tests():
    """
    Runs a test suite for KNN classifiers and saves the results to a JSON file.
    """
    # --- 1. Experiment Configuration ---
    print("Configuring the experiment...")
    RANDOM_STATE = 0
    N_SAMPLES = np.random.randint(500, 2000)
    N_FEATURES = np.random.randint(5, 50)
    N_INFORMATIVE = np.random.randint(1, N_FEATURES // 2 + 1)
    N_CLASSES = np.random.randint(2, 10)

    # --- 2. Data Generation & Splitting ---
    print(
        f"Generating a synthetic dataset with {N_SAMPLES} samples and {N_CLASSES} classes..."
    )
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # --- 3. Define Experiments ---
    classifiers_to_test = {}
    optimization_metrics = [
        "dissimilarity",
        "silhouette",
        "spread",
        "convex_hull_inter",
        "convex_hull_intra",
        "opposite_hyperplane",
    ]
    support_sample_methods = ["hnbf", "margin_clustering", "gabriel_graph"]

    for metric in optimization_metrics:
        for method in support_sample_methods:
            classifiers_to_test[f"Parameterless KNN ({metric}, {method})"] = (
                ParameterlessKNNClassifier(
                    metric=metric, support_samples_method=method, n_optimizer_calls=25
                )
            )
    classifiers_to_test["Scikit-learn KNN Wrapper"] = SklearnKNNParameterlessWrapper(
        n_optimizer_calls=25
    )

    # --- 4. Run All Experiments & Collect Results ---
    all_results = {}
    for name, classifier in classifiers_to_test.items():
        results = run_single_experiment(
            classifier, X_train, X_test, y_train, y_test, N_CLASSES, name
        )
        all_results[name] = results

    # --- 5. Save Results to JSON file ---
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_path = os.path.join(output_dir, "test_results.json")

    print("\n" + "=" * 60)
    print(f"Saving all test results to {output_path}")
    print("=" * 60)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("Successfully saved results.")


if __name__ == "__main__":
    run_all_tests()
