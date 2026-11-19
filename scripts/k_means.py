
import itertools
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os

os.makedirs("results", exist_ok=True)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

X_df = pd.read_csv("data/kimore_feature_matrix.csv")
metadata = pd.read_csv("data/kimore_metadata.csv")

# Validate dataset
assert len(X_df) == len(metadata)

X_df = X_df.replace([np.inf, -np.inf], np.nan)

if X_df.isna().any().any():
    raise ValueError("Missing values found in feature matrix!")

# Remove constant features
constant_cols = X_df.columns[
    X_df.nunique() <= 1
]

X_df = X_df.drop(columns=constant_cols)

print("Dataset shape:", X_df.shape)
print("Constant features removed:", len(constant_cols))

# Original labels: used ONLY for external evaluation
y_true = metadata["condition"].to_numpy()

# ==========================================
# 2. STANDARDIZATION
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

print("Scaled dataset:", X_scaled.shape)

# ==========================================
# 3. HYPERPARAMETER GRID
# ==========================================

param_grid = {
    "n_clusters": [2, 3, 4, 5, 6],
    "init": ["k-means++", "random"],
    "n_init": [10, 20],
    "max_iter": [300]
}

param_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in itertools.product(*param_grid.values())
]

print("Total combinations:", len(param_combinations))

# ==========================================
# 4. MLFLOW CONFIGURATION
# ==========================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "KIMORE_KMeans_Optimization"
)

# ==========================================
# 5. HYPERPARAMETER OPTIMIZATION
# ==========================================

best_score = -1
best_params = None
best_model = None

results = []

with mlflow.start_run(
    run_name="KMeans_Parent_Optimization"
):

    mlflow.log_param("dataset", "KIMORE")
    mlflow.log_param("samples", len(X_scaled))
    mlflow.log_param("features", X_scaled.shape[1])
    mlflow.log_param("scaling", "StandardScaler")
    mlflow.log_param(
        "selection_metric", "silhouette_score"
    )

    mlflow.log_param(
        "total_combinations",
        len(param_combinations)
    )

    for i, params in enumerate(param_combinations):

        with mlflow.start_run(
            run_name=f"Trial_{i+1}",
            nested=True
        ):

            model = KMeans(
                **params,
                random_state=42
            )

            cluster_labels = model.fit_predict(X_scaled)

            n_found = len(np.unique(cluster_labels))

            # Internal clustering metrics
            sil = silhouette_score(
                X_scaled, cluster_labels
            )

            db = davies_bouldin_score(
                X_scaled, cluster_labels
            )

            ch = calinski_harabasz_score(
                X_scaled, cluster_labels
            )

            # External evaluation only
            ari = adjusted_rand_score(
                y_true, cluster_labels
            )

            nmi = normalized_mutual_info_score(
                y_true, cluster_labels
            )

            ami = adjusted_mutual_info_score(
                y_true, cluster_labels
            )

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("random_state", 42)

            # Log metrics
            mlflow.log_metric("silhouette_score", sil)
            mlflow.log_metric("davies_bouldin_index", db)
            mlflow.log_metric("calinski_harabasz_score", ch)
            mlflow.log_metric("num_clusters_found", n_found)
            mlflow.log_metric("inertia", model.inertia_)

            # External validation metrics
            mlflow.log_metric("ARI", ari)
            mlflow.log_metric("NMI", nmi)
            mlflow.log_metric("AMI", ami)

            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )

            results.append({
                **params,
                "silhouette_score": sil,
                "davies_bouldin_index": db,
                "calinski_harabasz_score": ch,
                "inertia": model.inertia_,
                "ARI": ari,
                "NMI": nmi,
                "AMI": ami
            })

            print(
                f"Trial {i+1}/20 | "
                f"K={params['n_clusters']} | "
                f"Silhouette={sil:.4f} | "
                f"DB={db:.4f} | "
                f"ARI={ari:.4f} | "
                f"NMI={nmi:.4f}"
            )

            # Select best using silhouette ONLY
            if sil > best_score:
                best_score = sil
                best_params = params.copy()
                best_model = model

    # Log best configuration
    mlflow.log_params({
        f"best_{k}": v
        for k, v in best_params.items()
    })

    mlflow.log_metric(
        "best_silhouette_score",
        best_score
    )

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model"
    )

# ==========================================
# 6. SAVE EXPERIMENT RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/kmeans_experiment_results.csv",
    index=False
)

print("\n" + "=" * 55)
print("K-MEANS OPTIMIZATION COMPLETED")
print("=" * 55)

print("Best Parameters:", best_params)
print("Best Silhouette Score:", round(best_score, 4))

print("\nTop 5 configurations:")

print(
    results_df.sort_values(
        "silhouette_score",
        ascending=False
    ).head(5).to_string(index=False)
)

print("\nResults saved successfully!")
