
import os
import warnings
import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    cross_val_score,
    ParameterGrid
)

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
EXPERIMENT_NAME = "KIMORE_XGBoost_Optimization"

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

X = X_df.to_numpy(dtype=np.float32)

y_original = metadata["condition"].to_numpy(dtype=str)
groups = metadata["patient_id"].to_numpy(dtype=str)

assert len(X) == len(y_original) == len(groups), (
    "Feature, label and patient counts do not match!"
)

# Encode class labels for XGBoost
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_original)

print("Feature matrix:", X.shape)
print("Number of patients:", len(np.unique(groups)))
print("Classes:", list(label_encoder.classes_))

print("\nClass distribution:")
print(pd.Series(y_original).value_counts())

# --------------------------------------------------
# 3. PATIENT-WISE TRAIN TEST SPLIT
# --------------------------------------------------

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

assert set(groups[train_idx]).isdisjoint(
    set(groups[test_idx])
), "Patient leakage detected!"

# --------------------------------------------------
# 4. HYPERPARAMETER GRID
# --------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

parameter_combinations = list(ParameterGrid(param_grid))

cv = GroupKFold(n_splits=5)

print("\nTotal configurations:", len(parameter_combinations))

# --------------------------------------------------
# 5. PARENT MLFLOW RUN
# --------------------------------------------------

results = []

with mlflow.start_run(
    run_name="XGBoost_Parent_Optimization"
):

    mlflow.log_param("algorithm", "XGBClassifier")
    mlflow.log_param("dataset", "KIMORE")
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("validation", "Patient-wise GroupKFold")
    mlflow.log_param("test_size", 0.20)

    mlflow.log_param("num_classes", len(label_encoder.classes_))

    # --------------------------------------------------
    # 6. HYPERPARAMETER OPTIMIZATION
    # --------------------------------------------------

    for i, params in enumerate(parameter_combinations, start=1):

        print(
            f"\nTrial {i}/{len(parameter_combinations)}"
            f" | Parameters: {params}"
        )

        model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
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
            mlflow.log_param("objective", "multi:softprob")
            mlflow.log_param("tree_method", "hist")

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

    # --------------------------------------------------
    # 7. SELECT BEST MODEL
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="cv_macro_f1_mean",
        ascending=False
    )

    results_df.to_csv(
        "results/xgboost_cv_results.csv",
        index=False
    )

    best_params = {
        key: results_df.iloc[0][key]
        for key in param_grid
    }

    best_params["n_estimators"] = int(
        best_params["n_estimators"]
    )

    best_params["max_depth"] = int(
        best_params["max_depth"]
    )

    for key in [
        "learning_rate",
        "subsample",
        "colsample_bytree"
    ]:
        best_params[key] = float(best_params[key])

    print("\nBest parameters:")
    print(best_params)

    # --------------------------------------------------
    # 8. TRAIN FINAL MODEL
    # --------------------------------------------------

    final_model = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_train, y_train)

    # --------------------------------------------------
    # 9. TEST EVALUATION
    # --------------------------------------------------

    y_pred = final_model.predict(X_test)

    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)

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
    print("FINAL XGBOOST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    report = classification_report(
        y_test_original,
        y_pred_original,
        labels=label_encoder.classes_,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------
    # 10. LOG FINAL METRICS
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

    labels = label_encoder.classes_

    cm = confusion_matrix(
        y_test_original,
        y_pred_original,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title("XGBoost - KIMORE Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    confusion_path = (
        "results/plots/xgboost_confusion_matrix.png"
    )

    plt.savefig(confusion_path, dpi=300)
    plt.close()

    mlflow.log_artifact(confusion_path)

    # --------------------------------------------------
    # 12. CLASSIFICATION REPORT
    # --------------------------------------------------

    report_path = (
        "results/reports/xgboost_classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)

    # --------------------------------------------------
    # 13. FEATURE IMPORTANCE
    # --------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": X_df.columns,
        "importance": final_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_path = (
        "results/xgboost_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    mlflow.log_artifact(importance_path)

    # --------------------------------------------------
    # 14. LOG MODEL TO MLFLOW
    # --------------------------------------------------

    mlflow.xgboost.log_model(
        xgb_model=final_model,
        name="xgboost_model"
    )

    print("\nAll results and model logged to MLflow!")

print("\nXGBoost experiment completed!")
