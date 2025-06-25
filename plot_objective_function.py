import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification

from dissimilarity_score.dissimilarity import dissimilarity
from knn import KNNClassifier
from silhouette_score.silhouette import silhouette
from spread_score.spread import spread


def plot_objective_function():
    """
    This script generates synthetic classification data, computes objective
    function scores over a grid of hyperparameters for a custom KNN classifier,
    and visualizes these scores as interactive 3D surface plots.

    The objective functions evaluated are:
    - Dissimilarity Score
    - Silhouette Score
    - Spread Score

    The hyperparameters tuned are:
    - h: The bandwidth parameter of the RBF kernel.
    - k: The number of nearest neighbors for the sparse RBF kernel.
    """
    # 1. Create random synthetic classification data
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 2. Create a suitable hyperparameter grid for the KNN classifier
    h_range = np.linspace(0.1, 2.0, 10)
    k_range = np.arange(5, 55, 5)
    dissimilarity_scores = np.zeros((len(h_range), len(k_range)))
    silhouette_scores = np.zeros((len(h_range), len(k_range)))
    spread_scores = np.zeros((len(h_range), len(k_range)))

    # 3. Iterate over the hyperparameter grid
    for i, h in enumerate(h_range):
        for j, k in enumerate(k_range):

            h = float(h)  # Ensure h is a float
            k = int(k)  # Ensure k is an integer

            print(f"Testing h={h:.2f}, k={k}")
            # Initialize and fit the custom KNN classifier
            model = KNNClassifier(h=h, k=k)
            model.fit(X, y)

            # 4. Run probability prediction to get the similarity space Q
            # The predict_proba method calculates and stores Q
            model.predict_proba(X)
            Q = model.Q_

            # 5. Compute each objective function over the similarity space
            dissimilarity_scores[i, j] = dissimilarity(Q, y, h, k)
            silhouette_scores[i, j] = silhouette(Q, y, h, k)
            spread_scores[i, j] = spread(Q, y, h, k)

    # 6. Plot each objective function as an independent interactive plot
    def create_surface_plot(scores, title):
        fig = go.Figure(data=[go.Surface(z=scores, x=k_range, y=h_range)])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="k (Number of Neighbors)",
                yaxis_title="h (Bandwidth)",
                zaxis_title="Objective Score",
            ),
            autosize=False,
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90),
        )
        fig.show()

    create_surface_plot(dissimilarity_scores, "Dissimilarity Score vs. Hyperparameters")
    create_surface_plot(silhouette_scores, "Silhouette Score vs. Hyperparameters")
    create_surface_plot(spread_scores, "Spread Score vs. Hyperparameters")


if __name__ == "__main__":
    plot_objective_function()
