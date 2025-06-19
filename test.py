# run_experiment.py

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from knn import KNNClassifier


def run_classification_experiment():
    """
    Runs a complete test experiment for the KNNClassifier.

    This function will:
    1. Generate a synthetic dataset.
    2. Split the dataset into training and testing sets.
    3. Train the KNNClassifier.
    4. Evaluate its performance using multiple metrics.
    5. Visualize the results with a confusion matrix.
    """
    # --- 1. Experiment Configuration ---
    print("Configuring the experiment...")
    # Dataset parameters
    RANDOM_STATE = 0
    N_SAMPLES = random.randint(1000, 3000)
    N_FEATURES = random.randint(10, 50)
    N_INFORMATIVE = random.randint(5, N_FEATURES)
    N_CLASSES = random.randint(2, 8)

    # Classifier hyperparameters
    H_BANDWIDTH = 2.0  # Bandwidth for RBF kernel
    K_NEIGHBORS = 15  # Neighbors for sparse kernel

    # --- 2. Data Generation ---
    print(
        f"Generating a synthetic dataset with {N_SAMPLES} samples and {N_CLASSES} classes..."
    )
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_classes=N_CLASSES,
        n_clusters_per_class=2,
        random_state=RANDOM_STATE,
    )

    # --- 3. Data Splitting ---
    print("Splitting data into training and testing sets (70/30 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # --- 4. Model Training ---
    print("\nInitializing and training the KNNClassifier...")
    print(
        f"Hyperparameters: h={H_BANDWIDTH}, k={K_NEIGHBORS}, use_support_samples=True"
    )
    classifier = KNNClassifier(h=H_BANDWIDTH, k=K_NEIGHBORS, use_support_samples=True)

    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.4f} seconds.")

    # Report how many reference samples are being used
    if hasattr(classifier, "X_ref_"):
        num_ref_samples = classifier.X_ref_.shape[0]
        print(
            f"Classifier is using {num_ref_samples} reference samples (out of "
            f"{len(X_train)} original training samples)."
        )

    # --- 5. Prediction and Evaluation ---
    print("\nMaking predictions on the test set...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.4f} seconds.")

    print("\n--- Classifier Performance ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=[f"Class {i}" for i in range(N_CLASSES)]
        )
    )

    # --- 6. Visualization ---
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Class {i}" for i in range(N_CLASSES)],
        yticklabels=[f"Class {i}" for i in range(N_CLASSES)],
    )
    plt.title("Confusion Matrix for KNNClassifier")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.show()


if __name__ == "__main__":
    run_classification_experiment()
