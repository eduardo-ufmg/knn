"""
This module provides a K-Nearest Neighbors classifier that is compatible with
scikit-learn and optimized for performance and memory efficiency.

It uses a sparse RBF kernel for similarity computation and can optionally
reduce the training set to only "support samples" for faster predictions.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from similarity_space.similarity_space import similarity_space
from sparse_rbf.sparse_multivariate_rbf_kernel import sparse_multivarite_rbf_kernel
from support_samples.support_samples import support_samples

__all__ = ["KNNClassifier"]


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A K-Nearest Neighbors classifier based on a sparse RBF kernel and similarity space.

    This classifier is designed for performance and memory efficiency. It can optionally
    use a data reduction technique based on support samples to speed up computations.

    For each test sample, the classifier assigns the label of the class that it has
    the highest total similarity to, based on a sparse RBF kernel.

    Parameters
    ----------
    h : float, default=1.0
        The bandwidth parameter (length scale) for the RBF kernel. This parameter
        controls the "width" of the Gaussian kernel. Must be a positive number.

    k : int, default=10
        The number of nearest neighbors to consider for each sample when computing
        the sparse RBF kernel. This is what makes the computation efficient, as
        it avoids calculating similarities to all points.

    use_support_samples : bool, default=True
        If True, the classifier will be trained on a reduced subset of the training
        data composed of "support samples". These are samples that lie on the
        boundaries between classes, which can lead to faster predictions with
        little to no loss in accuracy.

    Attributes
    ----------
    X_fit_ : np.ndarray
        The training data that the classifier is fitted on. If `use_support_samples`
        is True, this will be the subset of support samples. Otherwise, it will
        be the full training set.

    y_fit_ : np.ndarray
        The labels corresponding to `X_fit_`.

    classes_ : np.ndarray of shape (n_classes,)
        The unique class labels seen during the `fit` process.
    """

    def __init__(self, h: float = 1.0, k: int = 10, use_support_samples: bool = True):
        self.h = h
        self.k = k
        self.use_support_samples = use_support_samples

    def fit(self, X, y):
        """
        Fit the K-NN classifier from the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # --- 1. Input Validation (Scikit-learn standard) ---
        X, y = check_X_y(X, y, accept_sparse=False)

        # Store the unique classes found in the training data
        self.classes_ = unique_labels(y)

        # --- 2. Data Reduction (Optional) ---
        # If enabled, find the support samples to reduce the size of the training set.
        if self.use_support_samples:
            self.X_fit_, self.y_fit_ = support_samples(X, y)
            # If support_samples returns an empty set (e.g., only one class is present),
            # fall back to using the full dataset to avoid errors.
            if self.X_fit_.shape[0] == 0:
                self.X_fit_, self.y_fit_ = X, y
        else:
            self.X_fit_, self.y_fit_ = X, y

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

        The probability of a sample belonging to a class is the normalized sum of
        similarities to all training samples of that class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        p : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples. Each row corresponds
            to a sample and contains the probabilities for each class in the
            order of `self.classes_`.
        """
        # --- 1. Check if the classifier has been fitted ---
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # --- 2. Compute Sparse Similarity Matrix ---
        # This computes the RBF kernel similarity between each sample in X and the
        # k-nearest neighbors in our fitted training data `self.X_fit_`.
        # The result S is a sparse matrix of shape (n_test_samples, n_fit_samples).
        S = sparse_multivarite_rbf_kernel(X, self.X_fit_, h=self.h, k=self.k)

        # --- 3. Transform to Similarity Space ---
        # This function sums the similarities for each class. The result Q is a
        # sparse matrix of shape (n_test_samples, n_classes).
        Q = similarity_space(S, self.y_fit_)

        # --- 4. Normalize to get probabilities ---
        Q_dense = Q.toarray()

        # Sum of similarities for each test sample across all classes
        q_sum = Q_dense.sum(axis=1, keepdims=True)

        # Handle cases where a test sample has zero similarity to all training samples
        # by assigning a uniform probability distribution across all classes.
        n_classes = len(self.classes_)
        # Use a small epsilon to avoid division by zero
        q_sum[q_sum == 0] = 1.0

        probabilities = Q_dense / q_sum

        # For rows that originally summed to zero, assign uniform probability
        zero_sum_mask = (q_sum == 1.0).flatten() & (Q_dense.sum(axis=1) == 0)
        if np.any(zero_sum_mask):
            probabilities[zero_sum_mask, :] = 1 / n_classes

        return probabilities

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        # --- 1. Get Probability Estimates ---
        probas = self.predict_proba(X)

        # --- 2. Find the most likely class ---
        # `np.argmax` returns the index of the class with the highest probability.
        max_indices = np.argmax(probas, axis=1)

        # --- 3. Map indices to class labels ---
        # `self.classes_` holds the actual labels in the correct order.
        y_pred = self.classes_[max_indices]

        return y_pred
