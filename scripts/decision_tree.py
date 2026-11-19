
import os
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/kimore_feature_matrix.csv"
METADATA_PATH = "data/kimore_metadata.csv"

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "KIMORE_DecisionTree_Optimization"

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading KIMORE dataset...")

X_df = pd.read_csv(DATA_PATH)
metadata = pd.read_csv(METADATA_PATH)

X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.loc[:, X_df.nunique() > 1]


# Convert all data to standard NumPy arrays
X = X_df.to_numpy(dtype=np.float64)

y = metadata["condition"].to_numpy(dtype=str)

groups = metadata["patient_id"].to_numpy(dtype=str)

assert len(X) == len(y) == len(groups), (
    "Feature, label and patient counts do not match!"
)

print("Feature matrix:", X.shape)
print("Number of patients:", len(np.unique(groups)))
print("\nClass distribution:")
print(pd.Series(y).value_counts())


# 3. PATIENT-WISE TRAIN TEST SPLIT


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

groups_train = groups[train_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training patients:",
    len(np.unique(groups[train_idx]))
)

print(
    "Testing patients:",
    len(np.unique(groups[test_idx]))
)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts())


# 4. HYPERPARAMETER GRID


param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2]
}

parameter_combinations = list(ParameterGrid(param_grid))

cv = GroupKFold(n_splits=5)

print("\nTotal configurations:", len(parameter_combinations))


# 5. MLflow EXPERIMENT


results = []

with mlflow.start_run(
    run_name="DecisionTree_Parent_Optimization"
):

    mlflow.log_param("algorithm", "DecisionTreeClassifier")
    mlflow.log_param("dataset", "KIMORE")
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("validation", "Patient-wise GroupKFold")
    mlflow.log_param("test_size", 0.20)

   
    # 6. HYPERPARAMETER OPTIMIZATION
   

    for i, params in enumerate(parameter_combinations, start=1):

        print(
            f"\nTrial {i}/{len(parameter_combinations)}"
            f" | Parameters: {params}"
        )

        model = DecisionTreeClassifier(
            **params,
            random_state=42,
            class_weight="balanced"
        )

        with mlflow.start_run(
            run_name=f"Trial_{i}",
            nested=True
        ):

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                groups=groups_train,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1
            )

            mean_score = scores.mean()
            std_score = scores.std()

            mlflow.log_params(params)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_metric("cv_macro_f1_mean", mean_score)
            mlflow.log_metric("cv_macro_f1_std", std_score)

            results.append({
                **params,
                "cv_macro_f1_mean": mean_score,
                "cv_macro_f1_std": std_score
            })

            print(
                f"Mean CV F1: {mean_score:.4f}"
                f" | Std: {std_score:.4f}"
            )
    # 7. SELECT BEST MODEL

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="cv_macro_f1_mean",
        ascending=False
    )

    results_df.to_csv(
        "results/decision_tree_cv_results.csv",
        index=False
    )

    best_params = {
        key: results_df.iloc[0][key]
        for key in param_grid
    }

    # Convert NumPy/Pandas values into Python values
    if pd.isna(best_params["max_depth"]):
        best_params["max_depth"] = None
    else:
        best_params["max_depth"] = int(best_params["max_depth"])

    best_params["min_samples_split"] = int(
        best_params["min_samples_split"]
    )
    best_params["min_samples_leaf"] = int(
        best_params["min_samples_leaf"]
    )

    best_params["criterion"] = str(
        best_params["criterion"]
    )

    print("\nBest parameters:")
    print(best_params)

  
    # 8. TRAIN FINAL MODEL
    

    final_model = DecisionTreeClassifier(
        **best_params,
        random_state=42,
        class_weight="balanced"
    )

    final_model.fit(X_train, y_train)

    
    # 9. TEST EVALUATION
    

    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("FINAL DECISION TREE RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------
    # 10. LOG FINAL RESULTS TO MLFLOW
    # --------------------------------------------------

    mlflow.log_params({
        "best_" + key: value
        for key, value in best_params.items()
    })

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision_macro", precision)
    mlflow.log_metric("test_recall_macro", recall)
    mlflow.log_metric("test_f1_macro", f1)

    # --------------------------------------------------
    # 11. CONFUSION MATRIX
    # --------------------------------------------------

    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Decision Tree - KIMORE Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    confusion_path = "results/plots/decision_tree_confusion_matrix.png"

    plt.savefig(confusion_path, dpi=300)
    plt.close()

    mlflow.log_artifact(confusion_path)

   
    # 12. CLASSIFICATION REPORT
    

    with open(
        "results/reports/decision_tree_classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    mlflow.log_artifact(
        "results/reports/decision_tree_classification_report.txt"
    )

    
    # 13. FEATURE IMPORTANCE
    

    importance_df = pd.DataFrame({
        "feature": X_df.columns,
        "importance": final_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df.to_csv(
        "results/decision_tree_feature_importance.csv",
        index=False
    )

    mlflow.log_artifact(
        "results/decision_tree_feature_importance.csv"
    )

   
    # 14. LOG TRAINED MODEL
    

    mlflow.sklearn.log_model(
        final_model,
        artifact_path="decision_tree_model",
        skops_trusted_types=[
        "sklearn.tree._tree.Tree"
        ]
    )

    print("\nSaved results:")
    print("results/decision_tree_cv_results.csv")
    print("results/decision_tree_feature_importance.csv")
    print("results/plots/decision_tree_confusion_matrix.png")
    print("results/reports/decision_tree_classification_report.txt")

print("\nDecision Tree experiment completed!")
