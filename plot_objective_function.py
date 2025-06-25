import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from dissimilarity_score.dissimilarity import dissimilarity
from knn import KNNClassifier
from silhouette_score.silhouette import silhouette
from spread_score.spread import spread

RANDOM_STATE = 0
N_SAMPLES = np.random.randint(100, 2001)
N_CLASSES = np.random.randint(2, 11)
N_FEATURES = np.random.randint(2, 41)
N_INFORMATIVE = np.random.randint(1, N_FEATURES)
N_REDUNDANT = N_FEATURES - N_INFORMATIVE - 1


def plot_objective_functions():
    """
    This function performs the following steps:
    1. Creates random synthetic classification data.
    2. Defines a hyperparameter grid for the KNN classifier.
    3. For each combination of hyperparameters, it predicts probabilities on the
       test data, which involves computing the similarity space (Q).
    4. It then computes three different objective functions (dissimilarity,
       silhouette, and spread) based on the similarity space.
    5. Finally, it plots these objective functions as interactive 3D surface
       plots, arranged side-by-side for comparison.
    """
    # 1. Create random synthetic classification data
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_classes=N_CLASSES,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Create a suitable hyperparameter grid for the KNN classifier
    h_values = np.linspace(0.1, 2.0, 10)
    k_values = np.arange(5, 16, 2)

    # Initialize arrays to store the results of the objective functions
    dissimilarity_scores = np.zeros((len(h_values), len(k_values)))
    silhouette_scores = np.zeros((len(h_values), len(k_values)))
    spread_scores = np.zeros((len(h_values), len(k_values)))

    # 3. Run probability prediction and compute objective functions
    for i, h in enumerate(h_values):
        for j, k in enumerate(k_values):
            h = float(h)  # Ensure h is a float
            k = int(k)  # Ensure k is an integer

            # Instantiate and fit the KNN classifier
            knn = KNNClassifier(h=h, k=k, use_support_samples=False)
            knn.fit(X_train, y_train)

            # predict_proba computes the similarity space Q
            knn.predict_proba(X_train)
            Q = knn.Q_

            # 4. Compute each objective function over the similarity space
            dissimilarity_scores[i, j] = dissimilarity(Q, y_train, h, k)
            silhouette_scores[i, j] = silhouette(Q, y_train, h, k)
            spread_scores[i, j] = spread(Q, y_train, h, k)

    # 5. Plot each objective function side by side as interactive plots
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Dissimilarity", "Silhouette", "Spread"),
    )

    # Create meshgrid for plotting
    K, H = np.meshgrid(k_values, h_values)

    # Add Dissimilarity Surface Plot
    fig.add_trace(
        go.Surface(z=dissimilarity_scores, x=K, y=H, colorscale="Viridis"),
        row=1,
        col=1,
    )

    # Add Silhouette Surface Plot
    fig.add_trace(
        go.Surface(z=silhouette_scores, x=K, y=H, colorscale="Plasma"),
        row=1,
        col=2,
    )

    # Add Spread Surface Plot
    fig.add_trace(
        go.Surface(z=spread_scores, x=K, y=H, colorscale="Inferno"),
        row=1,
        col=3,
    )

    fig.update_layout(
        title_text="Objective Function Landscapes",
        height=600,
        width=1200,
        scene1=dict(
            xaxis_title="k (neighbors)",
            yaxis_title="h (bandwidth)",
            zaxis_title="Dissimilarity",
        ),
        scene2=dict(
            xaxis_title="k (neighbors)",
            yaxis_title="h (bandwidth)",
            zaxis_title="Silhouette",
        ),
        scene3=dict(
            xaxis_title="k (neighbors)",
            yaxis_title="h (bandwidth)",
            zaxis_title="Spread",
        ),
    )

    fig.show()


if __name__ == "__main__":
    plot_objective_functions()
