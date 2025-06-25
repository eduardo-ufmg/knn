# K-Nearest Neighbors (KNN) Project

This repository contains a comprehensive implementation of the K-Nearest Neighbors (KNN) algorithm and its variations. It includes tools for parameterless KNN, dissimilarity scoring, silhouette scoring, and more. The project is designed for experimentation and evaluation of KNN-based methods on various datasets.

## Features

- **Custom KNN Implementation**: A Python-based implementation of the KNN algorithm.
- **Parameterless KNN**: A variation of KNN that eliminates the need for manual parameter tuning.
- **Dissimilarity Scoring**: Tools to compute dissimilarity scores between data points.
- **Silhouette Scoring**: Evaluate clustering quality using silhouette scores.
- **Dataset Management**: Scripts to store and manage datasets in Parquet format.
- **Extensive Experimentation**: Predefined experiments on multiple datasets with results stored in JSON format.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/eduardo-ufmg/knn.git
   cd knn
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments

To run the experiments, execute the `experiments.py` script:
```bash
python experiments.py
```

### Plotting Objective Functions

Use the `plot_objective_function.py` script to visualize objective functions:
```bash
python plot_objective_function.py
```
```

## Project Structure

- `knn.py`: Core implementation of the KNN algorithm.
- `parameterless_knn.py`: Implementation of parameterless KNN.
- `dissimilarity_score/`: Tools for computing dissimilarity scores.
- `silhouette_score/`: Scripts for silhouette scoring.
- `results/`: Contains results of experiments in JSON format.
- `sets/`: Datasets in Parquet format.
- `store_datasets/`: Scripts for dataset management.

## Datasets

The `sets/` directory contains various datasets in Parquet format, including:
- Abalone
- Car Evaluation
- Ionosphere
- Iris
- MNIST 784
- Mushroom
- Phoneme
- Seeds
- Wine
- Yeast

## Results

Experiment results are stored in the `results/` directory, organized by dataset and method (e.g., `custom_knn.json`, `sklearn_knn.json`).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Contact

For questions or suggestions, please contact [eduardohbc@ufmg.br].
