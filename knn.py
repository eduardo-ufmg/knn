from numbers import Integral

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

    Parameters:
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

    Attributes:
    classes_ : ndarray of shape (n_classes,)
        The unique class labels seen during `fit`.

    X_ref_ : ndarray of shape (n_references, n_features)
        The reference samples used for prediction. This will be the support
        samples if `use_support_samples` is True and support samples are found,
        otherwise it's the full training set.

    y_ref_ : ndarray of shape (n_references,)
        The labels for the reference samples.

    Q_ : ndarray of shape (n_samples, n_classes)
        The similarity space matrix, where each row corresponds to a sample and
        each column corresponds to a class. The values represent the
        similarity of the sample to each class based on the reference samples.
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

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns:
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        if self.h <= 0:
            raise ValueError("Bandwidth parameter h must be positive.")
        if not isinstance(self.k, Integral) or self.k <= 0:
            raise ValueError("Number of neighbors k must be a positive integer.")

        self.classes_ = unique_labels(y)

        if self.use_support_samples:
            self.X_ref_, self.y_ref_ = support_samples(X, y)
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

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns:
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered
            lexicographically as in `self.classes_`.
        """
        check_is_fitted(self)
        X = check_array(X)

        k = min(self.k, self.X_ref_.shape[0])
        kernel_matrix = sparse_multivarite_rbf_kernel(X, self.X_ref_, h=self.h, k=k)

        # Calculate the similarity space matrix. By passing `classes=self.classes_`,
        # we ensure the output matrix `Q` has a column for every class seen
        # during training, in the correct order.
        Q = similarity_space(kernel_matrix, self.y_ref_, classes=self.classes_)

        self.Q_ = Q

        row_sums = Q.sum(axis=1, keepdims=True)

        n_classes = len(self.classes_)
        uniform_prob = 1.0 / n_classes

        probabilities = np.full(Q.shape, uniform_prob)
        np.divide(Q, row_sums, out=probabilities, where=row_sums > 0)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns:
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        probabilities = self.predict_proba(X)
        max_prob_indices = np.argmax(probabilities, axis=1)

        return self.classes_[max_prob_indices]
