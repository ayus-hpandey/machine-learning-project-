
import pandas as pd
import numpy as np

# Load extracted dataset
df = pd.read_csv("data/kimore_extracted_features.csv")

print("=" * 55)
print("KIMORE FEATURE DATASET VALIDATION")
print("=" * 55)

# Separate metadata and features
metadata_cols = [
    "recording_id",
    "dataset_group",
    "condition",
    "patient_id",
    "exercise_id"
]

metadata = df[metadata_cols].copy()
X = df.drop(columns=metadata_cols)

# Dataset information
print("\nDataset shape:", df.shape)
print("Feature matrix:", X.shape)
print("Metadata shape:", metadata.shape)
print("\nMissing values:", X.isna().sum().sum())

print(
    "Infinite values:",
    np.isinf(X.to_numpy()).sum()
)

print(
    "Duplicate recordings:",
    metadata["recording_id"].duplicated().sum()
)

# Constant features
constant_features = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

print("\nConstant features:", len(constant_features))

# Duplicate feature columns
duplicate_features = X.columns[
    X.T.duplicated()
].tolist()

print("Duplicate feature columns:", len(duplicate_features))

# Feature statistics
print("\nFeature statistics:")
print(X.describe().T.head(15))

# Condition distribution
print("\nCondition distribution:")
print(metadata["condition"].value_counts())

# Exercise distribution
print("\nExercise distribution:")
print(metadata["exercise_id"].value_counts())

# Patient distribution
print("\nUnique patients:", metadata["patient_id"].nunique())

# Save separated datasets
metadata.to_csv("data/kimore_metadata.csv", index=False)
X.to_csv("data/kimore_feature_matrix.csv", index=False)

print("\nSaved:")
print("data/kimore_metadata.csv")
print("data/kimore_feature_matrix.csv")

print("\nValidation completed!")
