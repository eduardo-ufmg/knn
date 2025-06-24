import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from knn import KNNClassifier
from parameterless_knn import ParameterlessKNNClassifier


def run_single_experiment(
    classifier, X_train, X_test, y_train, y_test, n_classes, experiment_name
):
    """
    Trains, evaluates, and visualizes a single classifier instance.

    Args:
        classifier: An unfitted classifier instance (e.g., KNNClassifier).
        X_train, X_test, y_train, y_test: The training and testing data.
        n_classes (int): The number of classes in the dataset.
        experiment_name (str): The name of the experiment for titles and logs.
    """
    print("\n" + "=" * 60)
    print(f"Running Experiment: {experiment_name}")
    print("=" * 60)

    # --- 1. Model Training ---
    print(f"Training classifier: {classifier.__class__.__name__}")
    if isinstance(classifier, ParameterlessKNNClassifier):
        print(f"-> Optimization Metric: '{classifier.metric}'")
        print(f"-> Optimizer calls: {classifier.n_optimizer_calls}")
    else:
        print(f"-> Hyperparameters: h={classifier.h}, k={classifier.k}")

    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.4f} seconds.")

    if hasattr(classifier, "X_ref_"):
        num_ref_samples = classifier.X_ref_.shape[0]
        print(
            f"Classifier is using {num_ref_samples} reference samples (out of "
            f"{len(X_train)} original training samples)."
        )

    # --- 2. Prediction and Evaluation ---
    print("\nMaking predictions on the test set...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.4f} seconds.")

    print("\n--- Classifier Performance ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    # Added zero_division=0 to prevent warnings if a class has no predicted samples
    report = classification_report(
        y_test,
        y_pred,
        target_names=[f"Class {i}" for i in range(n_classes)],
        zero_division=0,
    )
    print(report)

    # --- 3. Visualization ---
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
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


def run_all_tests():
    """
    Runs a complete test suite for all implemented KNN classifiers.
    """
    # --- 1. Experiment Configuration (Fixed for reproducibility) ---
    print("Configuring the experiment...")
    RANDOM_STATE = 0
    N_SAMPLES = 1000
    N_FEATURES = 20
    N_INFORMATIVE = 10
    N_CLASSES = 4

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
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # --- 3. Define Experiments ---
    # A dictionary mapping experiment names to classifier instances.
    classifiers_to_test = {
        "Original KNN (h=2.0, k=15)": KNNClassifier(
            h=2.0, k=15, use_support_samples=True
        ),
        "Parameterless KNN (dissimilarity)": ParameterlessKNNClassifier(
            metric="dissimilarity", n_optimizer_calls=25
        ),
        "Parameterless KNN (silhouette)": ParameterlessKNNClassifier(
            metric="silhouette", n_optimizer_calls=25
        ),
        "Parameterless KNN (spread)": ParameterlessKNNClassifier(
            metric="spread", n_optimizer_calls=25
        ),
    }

    # --- 4. Run All Experiments ---
    for name, classifier in classifiers_to_test.items():
        run_single_experiment(
            classifier, X_train, X_test, y_train, y_test, N_CLASSES, name
        )


if __name__ == "__main__":
    run_all_tests()
