# knn/knn.py

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from similarity_space.similarity_space import similarity_space
from sparse_rbf.sparse_multivariate_rbf_kernel import sparse_multivarite_rbf_kernel
from support_samples.support_samples import support_samples


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A K-Nearest Neighbors classifier based on a sparse RBF kernel and a
    similarity space.

    This classifier is fully compatible with the Scikit-learn API. It assigns
    to each sample the label of the class with the highest similarity, which is
    computed by summing RBF kernel values over reference samples of the same
    class.

    For efficiency, the RBF kernel is computed sparsely, considering only the `k`
    nearest reference samples. Additionally, the set of reference samples can be
    reduced to only "support samples" that lie on class boundaries.

    Parameters
    ----------
    h : float, default=1.0
        The bandwidth parameter (length scale) of the RBF kernel. Must be
        positive.

    k : int, default=10
        The number of nearest neighbors to use for the sparse RBF kernel
        computation.

    use_support_samples : bool, default=True
        If True, the training data is pre-processed to find support samples,
        which are then used as the reference set for predictions. This can
        improve performance and memory efficiency.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels seen during `fit`.

    X_ref_ : ndarray of shape (n_references, n_features)
        The reference samples used for prediction. This will be the support
        samples if `use_support_samples` is True and support samples are found,
        otherwise it's the full training set.

    y_ref_ : ndarray of shape (n_references,)
        The labels for the reference samples.
    """

    def __init__(self, h: float = 1.0, k: int = 10, use_support_samples: bool = True):
        self.h = h
        self.k = k
        self.use_support_samples = use_support_samples

    def fit(self, X, y):
        """
        Fit the KNN classifier from the training dataset.

        If `use_support_samples` is True, this method will identify support
        samples from the training data to use as references for prediction.

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
        # Validate input data and parameters
        X, y = check_X_y(X, y)
        if self.h <= 0:
            raise ValueError("Bandwidth parameter h must be positive.")
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("Number of neighbors k must be a positive integer.")

        # Store the unique classes found in the training data
        self.classes_ = unique_labels(y)

        # Optionally, reduce the training data to only support samples
        if self.use_support_samples:
            self.X_ref_, self.y_ref_ = support_samples(X, y)
            # If no support samples are found (e.g., classes are well-separated),
            # fall back to using the full training set as a reference.
            if self.X_ref_.shape[0] == 0:
                self.X_ref_ = X
                self.y_ref_ = y
        else:
            self.X_ref_ = X
            self.y_ref_ = y

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for the test data X.

        The probability of a sample belonging to a class is calculated as its
        normalized similarity to that class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered
            lexicographically as in `self.classes_`.
        """
        check_is_fitted(self)
        X = check_array(X)

        # 1. Compute the sparse RBF kernel matrix between test samples (X) and
        # reference samples (self.X_ref_).
        # Ensure k is not larger than the number of available reference points.
        k = min(self.k, self.X_ref_.shape[0])
        kernel_matrix = sparse_multivarite_rbf_kernel(X, self.X_ref_, h=self.h, k=k)

        # 2. Calculate the similarity space matrix. This sums the kernel values
        # for each class, resulting in a matrix where each column represents
        # the total similarity to a class present in the reference set.
        q_ref = similarity_space(kernel_matrix, self.y_ref_)

        # The similarity_space function only returns columns for classes present
        # in `self.y_ref_`. We need to map these to the full set of classes
        # seen during `fit`.
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        Q = np.zeros((n_samples, n_classes))

        # Get the classes present in the reference set
        ref_classes = np.unique(self.y_ref_)

        # Find the column indices in the final Q matrix that correspond to the
        # classes in the reference set.
        class_indices = np.searchsorted(self.classes_, ref_classes)
        Q[:, class_indices] = q_ref

        # 3. Normalize the similarity scores to get probabilities.
        # Handle cases where a sample has zero similarity to all classes to
        # avoid division by zero.
        row_sums = Q.sum(axis=1, keepdims=True)

        # Default to uniform probability for zero-similarity samples
        uniform_prob = 1.0 / n_classes

        # Normalize rows with non-zero sums
        probabilities = np.full(Q.shape, uniform_prob)
        np.divide(Q, row_sums, out=probabilities, where=row_sums > 0)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Get the probability estimates
        probabilities = self.predict_proba(X)

        # Find the index of the class with the highest probability for each sample
        max_prob_indices = np.argmax(probabilities, axis=1)

        # Map these indices back to the actual class labels
        return self.classes_[max_prob_indices]
