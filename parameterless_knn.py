import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Use scikit-optimize for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from dissimilarity_score.dissimilarity import dissimilarity
from silhouette_score.silhouette import silhouette
from similarity_space.similarity_space import similarity_space
from sparse_rbf.sparse_multivariate_rbf_kernel import sparse_multivarite_rbf_kernel
from spread_score.spread import spread
from support_samples.support_samples import support_samples


class ParameterlessKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A parameter-less K-Nearest Neighbors classifier that uses
    Bayesian Optimization to find optimal hyperparameters.

    Parameters:
    metric : str, default='dissimilarity'
        The metric to optimize. Options: "dissimilarity", "silhouette", "spread".

    n_optimizer_calls : int, default=25
        The number of evaluations for the Bayesian optimizer. More calls can lead
        to better parameters but increase fitting time.

    Attributes:
    h_ : float
        The best bandwidth parameter (length scale) found for the RBF kernel.

    k_ : int
        The best number of nearest neighbors found for the sparse RBF kernel.
    ...
    """

    def __init__(self, metric: str = "dissimilarity", n_optimizer_calls: int = 25):
        if metric not in ["dissimilarity", "silhouette", "spread"]:
            raise ValueError(
                f"Invalid metric '{metric}'. Choose from 'dissimilarity', "
                "'silhouette', or 'spread'."
            )
        self.metric = metric
        self.n_optimizer_calls = n_optimizer_calls

    def fit(self, X, y):
        """
        Fit the classifier using Bayesian optimization to find hyperparameters.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.metric_ = self.metric

        metric_functions = {
            "dissimilarity": dissimilarity,
            "silhouette": silhouette,
            "spread": spread,
        }
        metric_func = metric_functions[self.metric_]

        X_ref, y_ref = support_samples(X, y)
        if X_ref.shape[0] < 2:
            self.X_ref_, self.y_ref_ = X, y
        else:
            self.X_ref_, self.y_ref_ = X_ref, y_ref

        n_ref_samples = self.X_ref_.shape[0]
        if n_ref_samples <= 1:
            self.h_, self.k_ = 1.0, 3
            self.is_fitted_ = True
            warnings.warn(
                "Not enough reference samples to optimize. Using defaults.", UserWarning
            )
            return self

        # --- Bayesian Optimization ---

        # 1. Define the search space for h and k
        space = [
            Real(low=1e-2, high=1e1, name="h"),
            Integer(low=1, high=max(1, n_ref_samples // 2), name="k"),
        ]

        # 2. Define the objective function to be minimized
        @use_named_args(space)
        def objective(**params):
            kernel_matrix = sparse_multivarite_rbf_kernel(
                self.X_ref_, self.X_ref_, **params
            )
            Q = similarity_space(kernel_matrix, self.y_ref_, classes=self.classes_)

            # Here we scale the factors

            score = metric_func(Q, self.y_ref_, **params)
            # We minimize the negative score because the optimizer finds minima
            return -score

        # 3. Run the optimizer
        # This context manager will temporarily catch and handle warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            res = gp_minimize(
                objective,
                space,
                n_calls=self.n_optimizer_calls,
                n_initial_points=10,
                random_state=0,
                verbose=False,
            )

        if res is not None:
            self.h_ = res.x[0]
            self.k_ = res.x[1]
        else:
            self.h_, self.k_ = 1.0, 3
            warnings.warn("Optimization failed. Using default parameters.", UserWarning)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X."""
        check_is_fitted(self)
        X = check_array(X)
        k = min(self.k_, self.X_ref_.shape[0])
        kernel_matrix = sparse_multivarite_rbf_kernel(X, self.X_ref_, h=self.h_, k=k)
        Q = similarity_space(kernel_matrix, self.y_ref_, classes=self.classes_)
        row_sums = Q.sum(axis=1, keepdims=True)
        n_classes = len(self.classes_)
        uniform_prob = 1.0 / n_classes
        probabilities = np.full(Q.shape, uniform_prob, dtype=np.float64)
        np.divide(Q, row_sums, out=probabilities, where=row_sums > 0)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels for the provided data."""
        probabilities = self.predict_proba(X)
        max_prob_indices = np.argmax(probabilities, axis=1)
        return self.classes_[max_prob_indices]
