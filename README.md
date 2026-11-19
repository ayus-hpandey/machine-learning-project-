# KiMoRe Movement Classification

This repository contains a Python machine-learning workflow for analyzing the
KiMoRe rehabilitation dataset. It extracts movement statistics from skeletal
joint-position recordings, evaluates supervised classifiers for the recording
condition, and explores unsupervised clustering. Training and experiment
metrics are tracked with MLflow.

## What the workflow does

1. **Extract features** from `JointPosition` CSV recordings under `KiMoRe/`.
   Each recording is expected to contain 100 values per frame (25 joints with
   four values each). The extractor uses the XYZ coordinates to calculate
   per-joint position, displacement, velocity, acceleration, and jerk
   statistics.
2. **Prepare the model inputs** by separating recording metadata from numeric
   features and saving the feature matrix and metadata as CSV files.
3. **Run experiments** using decision tree, random forest, and XGBoost
   classifiers, or K-means and agglomerative clustering. Classifier evaluation
   uses patient-grouped train/test and cross-validation splits to avoid placing
   the same patient in both training and test data.

Prepared feature and metadata CSVs are already included in `data/`, so the
experiments can  run without repeating feature extraction.

## Repository layout

```text
KiMoRe/                  Raw KiMoRe recordings and related source files
data/
  kimore_extracted_features.csv
  kimore_feature_matrix.csv
  kimore_metadata.csv
  failed_recordings.csv
scripts/
  extract_all_features.py
  validate_features.py
  decision_tree.py
  random_forest.py
  xgboost_classifier.py
  k_means.py
  heirarchial_clustering.py
results/                  Experiment reports, plots, and tabular results
requirements.txt          Pinned Python dependencies
mlflow.db, mlruns/        MLflow tracking data and artifacts
```

## Setup

Use Python with `pip`. From the repository root, create and activate a virtual
environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The XGBoost experiment imports `xgboost`, which is not listed in
`requirements.txt`. To run that experiment, install it separately:

```bash
python -m pip install xgboost
```

## Run the workflow

Run commands from the repository root; the scripts use repository-relative
paths.

### Regenerate the prepared data

```bash
python scripts/extract_all_features.py
python scripts/validate_features.py
```

Feature extraction writes `data/kimore_extracted_features.csv` and records
unreadable or invalid recordings in `data/failed_recordings.csv`. Validation
creates `data/kimore_metadata.csv` and `data/kimore_feature_matrix.csv`.
Running these commands replaces those output files.

### Run a supervised experiment

```bash
python scripts/decision_tree.py
python scripts/random_forest.py
python scripts/xgboost_classifier.py
```

Each classifier performs a parameter search with patient-wise grouped
cross-validation, evaluates the selected model on a held-out patient-wise split,
and logs its runs to MLflow. The scripts save classification reports,
confusion-matrix plots, feature importances, and search results under
`results/`.

### Run a clustering experiment

```bash
python scripts/k_means.py
python scripts/heirarchial_clustering.py
```

Both scripts standardize the feature matrix and compare cluster configurations.
They report internal clustering metrics and compare clusters with the known
condition labels using external metrics. The result tables are written to
`results/`.

### View MLflow runs

After running an experiment, start the MLflow UI from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the address printed by MLflow (by default, `http://127.0.0.1:5000`) to
browse experiments, parameters, metrics, and artifacts.

## Notes

- The feature extraction script requires the raw `KiMoRe/` directory and only
  processes files whose names contain `JointPosition`.
- The classifier scripts expect `data/kimore_feature_matrix.csv` and
  `data/kimore_metadata.csv` to have matching row order and length.
- Model training and hyperparameter searches can take a while, especially for
  the random forest experiment.
- The scripts use `results/` and `mlflow.db` for generated outputs and tracking
  data.
