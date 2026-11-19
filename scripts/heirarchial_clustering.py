
import os
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/kimore_feature_matrix.csv"
METADATA_PATH = "data/kimore_metadata.csv"

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "KIMORE_Hierarchical_Optimization"

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading KIMORE dataset...")

X_df = pd.read_csv(DATA_PATH)
metadata = pd.read_csv(METADATA_PATH)

# Remove any constant features, if present
X_df = X_df.loc[:, X_df.nunique() > 1]

X = X_df.select_dtypes(include=[np.number]).values

# Keep condition labels separate from clustering features
y_true = metadata["condition"].values

print("Feature matrix shape:", X.shape)
print("Metadata shape:", metadata.shape)
print("Number of features:", X.shape[1])
print("Number of recordings:", len(X))

assert len(X) == len(y_true), "Feature and metadata rows do not match!"

# --------------------------------------------------
# 3. STANDARDIZATION
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Standardized data shape:", X_scaled.shape)

# --------------------------------------------------
# 4. HYPERPARAMETER GRID
# --------------------------------------------------

cluster_values = [2, 3, 4, 5, 6]
linkage_values = ["ward", "complete", "average", "single"]

results = []

total_trials = len(cluster_values) * len(linkage_values)
trial_number = 0

# --------------------------------------------------
# 5. PARENT MLflow RUN
# --------------------------------------------------

with mlflow.start_run(run_name="Hierarchical_Parent_Optimization"):

    mlflow.log_param("algorithm", "AgglomerativeClustering")
    mlflow.log_param("dataset", "KIMORE")
    mlflow.log_param("n_samples", X.shape[0])
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("scaling", "StandardScaler")

    # --------------------------------------------------
    # 6. GRID SEARCH
    # --------------------------------------------------

    for n_clusters in cluster_values:

        for linkage in linkage_values:

            trial_number += 1

            print(
                f"\nTrial {trial_number}/{total_trials}"
                f" | K={n_clusters}"
                f" | Linkage={linkage}"
            )

            metric = "euclidean"

            with mlflow.start_run(
                run_name=f"Trial_{trial_number}_K{n_clusters}_{linkage}",
                nested=True
            ):

                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    metric=metric
                )

                cluster_labels = model.fit_predict(X_scaled)

                # Internal evaluation
                silhouette = silhouette_score(
                    X_scaled, cluster_labels
                )

                db_score = davies_bouldin_score(
                    X_scaled, cluster_labels
                )

                ch_score = calinski_harabasz_score(
                    X_scaled, cluster_labels
                )

                # External evaluation
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
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_param("linkage", linkage)
                mlflow.log_param("metric", metric)

                # Log metrics
                mlflow.log_metric("silhouette_score", silhouette)
                mlflow.log_metric("davies_bouldin_score", db_score)
                mlflow.log_metric("calinski_harabasz_score", ch_score)

                mlflow.log_metric("ari", ari)
                mlflow.log_metric("nmi", nmi)
                mlflow.log_metric("ami", ami)

                # Store results
                results.append({
                    "n_clusters": n_clusters,
                    "linkage": linkage,
                    "metric": metric,
                    "silhouette": silhouette,
                    "davies_bouldin": db_score,
                    "calinski_harabasz": ch_score,
                    "ari": ari,
                    "nmi": nmi,
                    "ami": ami
                })

                print(
                    f"Silhouette={silhouette:.4f}"
                    f" | DB={db_score:.4f}"
                    f" | CH={ch_score:.2f}"
                    f" | ARI={ari:.4f}"
                    f" | NMI={nmi:.4f}"
                    f" | AMI={ami:.4f}"
                )

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/hierarchical_experiment_results.csv",
    index=False
)

# --------------------------------------------------
# 8. BEST CONFIGURATIONS
# --------------------------------------------------

print("\n" + "=" * 65)
print("HIERARCHICAL CLUSTERING RESULTS")
print("=" * 65)

print("\nBest configuration by Silhouette Score:")
print(
    results_df.loc[
        results_df["silhouette"].idxmax()
    ].to_string()
)

print("\nBest configuration by ARI:")
print(
    results_df.loc[
        results_df["ari"].idxmax()
    ].to_string()
)

print("\nBest configuration by NMI:")
print(
    results_df.loc[
        results_df["nmi"].idxmax()
    ].to_string()
)

print("\nBest configuration by Davies-Bouldin Index:")
print(
    results_df.loc[
        results_df["davies_bouldin"].idxmin()
    ].to_string()
)

print("\nAll results saved to:")
print("results/hierarchical_experiment_results.csv")

print("\nHierarchical clustering experiments completed!")
