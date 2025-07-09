import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from CorrelationFilter.CorrelationFilter import CorrelationFilter
from parameterless_knn import ParameterlessKNNClassifier
from sklearn_knn_parameterless_wrapper import SklearnKNNParameterlessWrapper

DATASETS_DIR = Path("sets/")
RESULTS_DIR = Path("results/")


def evaluate_model(dataset_path, model_name, model, n_splits=10, n_opt_calls=10):
    """
    Load data, run CV for one model on one dataset, and return a dict of results
    and the list of fold accuracies.
    Any exception bubbles up, and will be caught in the main loop.
    """
    dataset = dataset_path.stem
    data = pd.read_parquet(dataset_path)
    X = data.drop("target", axis=1).values
    y = data["target"].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_metrics = {"accuracies": [], "train_times": [], "predict_times": []}

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("varianceth", VarianceThreshold(0.1)),
                ("corr", CorrelationFilter(0.9)),
                ("pca", PCA(n_components=0.9)),
            ]
        )
        X_tr_s = pipe.fit_transform(X_tr)
        X_te_s = pipe.transform(X_te)

        # train
        t0 = time.perf_counter()
        model.fit(X_tr_s, y_tr)
        fold_metrics["train_times"].append(time.perf_counter() - t0)

        # predict
        t0 = time.perf_counter()
        y_pred = model.predict(X_te_s)
        fold_metrics["predict_times"].append(time.perf_counter() - t0)

        fold_metrics["accuracies"].append(accuracy_score(y_te, y_pred))

    result = {
        "model_name": model_name,
        "dataset": dataset,
        "hyperparameters": model.get_params(),
        "metrics": {
            "accuracy_mean": float(np.mean(fold_metrics["accuracies"])),
            "accuracy_std": float(np.std(fold_metrics["accuracies"])),
            "train_time_mean": float(np.mean(fold_metrics["train_times"])),
            "train_time_std": float(np.std(fold_metrics["train_times"])),
            "predict_time_mean": float(np.mean(fold_metrics["predict_times"])),
            "predict_time_std": float(np.std(fold_metrics["predict_times"])),
        },
    }
    return result, fold_metrics["accuracies"]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    dataset_paths = sorted(DATASETS_DIR.glob("*.parquet"))
    if not dataset_paths:
        raise RuntimeError("No datasets found!")

    # 1) Define all model configurations
    model_configs = []
    metrics = [
        "dissimilarity",
        "silhouette",
        "spread",
        "convex_hull_inter",
        "convex_hull_intra",
        "opposite_hyperplane",
        "accuracy",
    ]
    ss_methods = ["hnbf", "margin_clustering", "gabriel_graph", "none"]
    for m in metrics:
        for ss in ss_methods:
            name = f"parameterless_knn_{m}_{ss}"
            # Use a factory lambda to create fresh instances later
            model_factory = lambda m=m, ss=ss: ParameterlessKNNClassifier(
                metric=m, support_samples_method=ss, n_optimizer_calls=10
            )
            model_configs.append((name, model_factory))

    # Add baseline model
    baseline_name = "sklearn_knn_wrapper"
    model_configs.append(
        (baseline_name, lambda: SklearnKNNParameterlessWrapper(n_optimizer_calls=10))
    )
    all_model_names = [name for name, _ in model_configs]

    # 2) Build (dataset, model) tasks
    tasks = []
    for ds in dataset_paths:
        for name, model_factory in model_configs:
            tasks.append((ds, name, model_factory()))

    # 3) Dispatch all tasks into one pool
    cpu_count = os.cpu_count()
    cpu = cpu_count // 2 if cpu_count else 1
    with ProcessPoolExecutor(max_workers=cpu) as exe:
        future_to_task = {
            exe.submit(evaluate_model, ds, nm, mdl): (ds.stem, nm)
            for ds, nm, mdl in tasks
        }

        # 4) As tasks finish, save results and aggregate accuracies
        all_accuracies: Dict[str, Dict[str, List[float]]] = {}
        for future in as_completed(future_to_task):
            ds_name, model_name = future_to_task[future]
            try:
                res, fold_accuracies = future.result()

                outdir = RESULTS_DIR / ds_name
                outdir.mkdir(exist_ok=True, parents=True)
                with open(outdir / f"{model_name}.json", "w") as f:
                    json.dump(res, f, indent=4)

                all_accuracies.setdefault(ds_name, {})[model_name] = fold_accuracies
                print(
                    f"[OK] {ds_name:15} · {model_name:30} → {res['metrics']['accuracy_mean']:.4f}"
                )

            except Exception as e:
                print(f"[ERROR] {ds_name:15} · {model_name:30} → {e}")

    # 5) Post-hoc statistical tests per dataset
    alpha = 0.05
    for ds_name, completed_models in all_accuracies.items():
        ref_accuracies = completed_models.get("sklearn_knn_wrapper")
        if not ref_accuracies:
            print(
                f"[STATS] {ds_name:15} · Reference model 'sklearn_knn_wrapper' missing. Skipping stats."
            )
            continue

        stats: List[Dict[str, Any]] = []
        # Iterate over all models that were supposed to run
        for model_name in all_model_names:
            if model_name == "sklearn_knn_wrapper":
                continue  # Don't compare the reference to itself

            # Check if the model completed successfully for this dataset
            model_accuracies = completed_models.get(model_name)

            if model_accuracies:
                # Success: perform the Wilcoxon test
                stat, p = wilcoxon(
                    model_accuracies, ref_accuracies, zero_method="zsplit"
                )
                stat, p = cast(float, stat), cast(float, p)

                mean_diff = np.mean(model_accuracies) - np.mean(ref_accuracies)
                conclusion = (
                    "equivalent"
                    if p > alpha
                    else ("better" if mean_diff > 0 else "worse")
                )

                stats.append(
                    {
                        "comparison": f"{model_name}_vs_sklearn",
                        "statistic": float(stat),
                        "p_value": float(p),
                        "conclusion": conclusion,
                    }
                )
            else:
                # Failure: report that the model's results are missing
                stats.append(
                    {
                        "comparison": f"{model_name}_vs_sklearn",
                        "statistic": None,
                        "p_value": None,
                        "conclusion": "failed",
                    }
                )

        # Write out the statistical comparison results
        stats_path = RESULTS_DIR / ds_name / "statistical_comparisons.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"[STATS] {ds_name:15} · Comparisons saved to {stats_path}")


if __name__ == "__main__":
    main()
