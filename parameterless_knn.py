import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Use scikit-optimize for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from convex_hull_inter.convex_hull_inter import convex_hull_inter
from convex_hull_intra.convex_hull_intra import convex_hull_intra
from dissimilarity_score.dissimilarity import dissimilarity
from opposite_hyperplane.opposite_hyperplane import opposite_hyperplane
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
    metric : str, default='accuracy'
        The metric to optimize. Options: "dissimilarity", "silhouette", "spread", "convex_hull_inter", "convex_hull_intra", "opposite_hyperplane", 'accuracy'.

    support_samples_method : str, default='none'
        The method to find support samples. Options: 'hnbf', 'margin_clustering', 'gabriel_graph', 'none'.

    n_optimizer_calls : int, default=10
        The number of evaluations for the Bayesian optimizer. More calls can lead
        to better parameters but increase fitting time.

    Attributes:
    h_ : float
        The best bandwidth parameter (length scale) found for the RBF kernel.

    k_ : int
        The best number of nearest neighbors found for the sparse RBF kernel.
    ...
    """

    def __init__(
        self,
        metric: str = "accuracy",
        support_samples_method: str = "none",
        n_optimizer_calls: int = 10,
    ):
        if metric not in [
            "dissimilarity",
            "silhouette",
            "spread",
            "convex_hull_inter",
            "convex_hull_intra",
            "opposite_hyperplane",
            "accuracy",
        ]:
            raise ValueError(
                f"Invalid metric '{metric}'. Choose from 'dissimilarity', "
                "'silhouette', 'spread', 'convex_hull_inter', "
                "'convex_hull_intra', 'opposite_hyperplane', 'accuracy'."
            )

        if support_samples_method not in [
            "hnbf",
            "margin_clustering",
            "gabriel_graph",
            "none",
        ]:
            raise ValueError(
                f"Invalid support_samples_method '{support_samples_method}'. "
                "Choose from 'hnbf', 'margin_clustering', 'gabriel_graph', 'none'."
            )

        if n_optimizer_calls <= 0:
            raise ValueError("n_optimizer_calls must be a positive integer.")

        self.metric = metric
        self.support_samples_method = support_samples_method
        self.n_optimizer_calls = n_optimizer_calls

    def fit(self, X, y):
        """
        Fit the classifier using Bayesian optimization to find hyperparameters.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.metric_ = self.metric

        X_ref, y_ref = support_samples(X, y, method=self.support_samples_method)
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
        h_min, h_max = 1e-2, 1e1
        k_min, k_max = 1, max(1, n_ref_samples // 2)
        space = [
            Real(low=h_min, high=h_max, name="h"),
            Integer(low=k_min, high=k_max, name="k"),
        ]

        # 2. Define the objective function to be minimized
        if self.metric_ in [
            "dissimilarity",
            "silhouette",
            "spread",
            "convex_hull_inter",
            "convex_hull_intra",
            "opposite_hyperplane",
        ]:

            metric_functions = {
                "dissimilarity": dissimilarity,
                "silhouette": silhouette,
                "spread": spread,
                "convex_hull_inter": convex_hull_inter,
                "convex_hull_intra": convex_hull_intra,
                "opposite_hyperplane": opposite_hyperplane,
            }
            metric_func = metric_functions[self.metric_]

            @use_named_args(space)
            def objective(**params):
                kernel_matrix = sparse_multivarite_rbf_kernel(
                    self.X_ref_, self.X_ref_, **params
                )
                Q = similarity_space(kernel_matrix, self.y_ref_, classes=self.classes_)

                factor_h = (float(params["h"]) - float(h_min)) / float(h_max - h_min)
                factor_k = (float(params["k"]) - float(k_min)) / float(k_max - k_min)

                # Pass self.classes_ to ensure the metric function is aware of all
                # original classes, even if some are missing from y_ref_.
                score = metric_func(
                    Q,
                    self.y_ref_,
                    factor_h=factor_h,
                    factor_k=factor_k,
                    classes=self.classes_,
                )
                # We minimize the negative score because the optimizer finds minima
                return -score

        elif self.metric_ == "accuracy":

            @use_named_args(space)
            def objective(**params):

                accuracies = []

                for train_index, test_index in StratifiedKFold(shuffle=True).split(
                    self.X_ref_, self.y_ref_
                ):
                    X_train, X_test = self.X_ref_[train_index], self.X_ref_[test_index]
                    y_train, y_test = self.y_ref_[train_index], self.y_ref_[test_index]

                    # Ensure k is not larger than the number of training samples
                    k_fold = min(params["k"], X_train.shape[0])
                    if k_fold == 0:
                        # Handle cases where the training fold is empty
                        accuracies.append(0)
                        continue

                    kernel_matrix = sparse_multivarite_rbf_kernel(
                        X_test, X_train, h=params["h"], k=k_fold
                    )
                    Q = similarity_space(kernel_matrix, y_train, classes=self.classes_)
                    row_sums = Q.sum(axis=1, keepdims=True)
                    n_classes = len(self.classes_)
                    uniform_prob = 1.0 / n_classes
                    probabilities = np.full(Q.shape, uniform_prob, dtype=np.float64)
                    np.divide(Q, row_sums, out=probabilities, where=row_sums > 0)
                    predictions = self.classes_[np.argmax(probabilities, axis=1)]
                    accuracies.append(accuracy_score(y_test, predictions))

                # Return the negative mean accuracy to minimize it
                return -np.mean(accuracies)

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
