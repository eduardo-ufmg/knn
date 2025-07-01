import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification

from convex_hull_inter.convex_hull_inter import convex_hull_inter
from convex_hull_intra.convex_hull_intra import convex_hull_intra
from dissimilarity_score.dissimilarity import dissimilarity
from knn import KNNClassifier
from opposite_hyperplane.opposite_hyperplane import opposite_hyperplane
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
    - Convex Hull Inter (Intersection of Convex Hulls)
    - Convex Hull Intra (Average n-volume of Convex Hulls)
    - Opposite Hyperplane (Cosine similarity of hyperplane normals)

    The hyperparameters tuned are:
    - h: The bandwidth parameter of the RBF kernel.
    - k: The number of nearest neighbors for the sparse RBF kernel.
    """
    # 1. Create random synthetic classification data
    n_features = np.random.randint(2, 10)
    X, y = make_classification(
        n_samples=np.random.randint(100, 1000),
        n_features=n_features,
        n_informative=np.random.randint(1, n_features),
        n_redundant=0,
        n_classes=np.random.randint(2, 5),
    )
    # 2. Create a suitable hyperparameter grid for the KNN classifier
    h_range = np.linspace(0.01, 10.0, 10)
    k_range = np.arange(5, 60, 10)
    dissimilarity_scores = np.zeros((len(h_range), len(k_range)))
    silhouette_scores = np.zeros((len(h_range), len(k_range)))
    spread_scores = np.zeros((len(h_range), len(k_range)))
    convex_hull_inter_scores = np.zeros((len(h_range), len(k_range)))
    convex_hull_intra_scores = np.zeros((len(h_range), len(k_range)))
    opposite_hyperplane_scores = np.zeros((len(h_range), len(k_range)))

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

            factor_h = (h - h_range.min()) / (h_range.max() - h_range.min())
            factor_k = (k - k_range.min()) / (k_range.max() - k_range.min())

            print(f"Scaled factors: factor_h={factor_h:.2f}, factor_k={factor_k:.2f}")

            # 5. Compute each objective function over the similarity space
            dissimilarity_scores[i, j] = dissimilarity(Q, y, factor_h, factor_k)
            silhouette_scores[i, j] = silhouette(Q, y, factor_h, factor_k)
            spread_scores[i, j] = spread(Q, y, factor_h, factor_k)
            convex_hull_inter_scores[i, j] = convex_hull_inter(Q, y, factor_h, factor_k)
            convex_hull_intra_scores[i, j] = convex_hull_intra(Q, y, factor_h, factor_k)
            opposite_hyperplane_scores[i, j] = opposite_hyperplane(
                Q, y, factor_h, factor_k
            )

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
    create_surface_plot(
        convex_hull_inter_scores, "Convex Hull Inter vs. Hyperparameters"
    )
    create_surface_plot(
        convex_hull_intra_scores, "Convex Hull Intra vs. Hyperparameters"
    )
    create_surface_plot(
        opposite_hyperplane_scores, "Opposite Hyperplane vs. Hyperparameters"
    )


if __name__ == "__main__":
    plot_objective_function()
