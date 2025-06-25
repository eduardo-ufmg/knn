import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Use scikit-optimize for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args


class SklearnKNNParameterlessWrapper(BaseEstimator, ClassifierMixin):
    """
    A parameter-less K-Nearest Neighbors classifier wrapper that uses
    Bayesian Optimization to find optimal hyperparameters for scikit-learn's
    KNeighborsClassifier.

    The optimization is based on the 10-fold cross-validation accuracy
    on the training data.

    Parameters:
    n_optimizer_calls : int, default=25
        The number of evaluations for the Bayesian optimizer. More calls can lead
        to better parameters but increase fitting time.

    Attributes:
    best_params_ : dict
        The dictionary of best hyperparameters found by the optimizer.

    knn_ : KNeighborsClassifier
        The fitted KNeighborsClassifier instance with the best hyperparameters.
    """

    def __init__(self, n_optimizer_calls: int = 25):
        self.n_optimizer_calls = n_optimizer_calls

    def fit(self, X, y):
        """
        Fit the classifier using Bayesian optimization to find the best
        hyperparameters for KNeighborsClassifier based on cross-validation accuracy.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # --- Bayesian Optimization ---

        # 1. Define the search space for n_neighbors, weights, and metric
        # The maximum number of neighbors is limited to half the number of samples
        max_neighbors = max(1, len(y) // 2)
        space = [
            Integer(1, max_neighbors, name="n_neighbors"),
            Categorical(["uniform", "distance"], name="weights"),
            Categorical(["euclidean", "manhattan", "minkowski"], name="metric"),
        ]

        # 2. Define the objective function to be minimized
        @use_named_args(space)
        def objective(**params):
            native_params = {
                "n_neighbors": int(params["n_neighbors"]),
                "weights": str(params["weights"]),
                "metric": str(params["metric"]),
            }

            knn = KNeighborsClassifier(**native_params)
            # Use 10-fold cross-validation
            score = np.mean(cross_val_score(knn, X, y, cv=10, scoring="accuracy"))
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
                n_initial_points=10,  # Start with 10 random points
                random_state=0,
                verbose=False,  # Set to True for detailed progress
            )

        if res is None:
            raise RuntimeError("Bayesian optimization failed to return a valid result.")

        # Store the best parameters found
        self.best_params_ = {
            "n_neighbors": int(res.x[0]),
            "weights": str(res.x[1]),
            "metric": str(res.x[2]),
        }

        # Fit the final model on the entire dataset with the best parameters
        self.knn_ = KNeighborsClassifier(**self.best_params_)
        self.knn_.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels for the provided data."""
        check_is_fitted(self)
        X = check_array(X)
        return self.knn_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X."""
        check_is_fitted(self)
        X = check_array(X)
        return np.ndarray(self.knn_.predict_proba(X))
