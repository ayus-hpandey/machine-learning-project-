
import os
import glob
import pandas as pd
import numpy as np

DATASET_PATH = r"KiMoRe"

files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*JointPosition*.csv"),
    recursive=True
)

file_path = files[0]

# Read CSV
df = pd.read_csv(
    file_path,
    header=None,
    skip_blank_lines=True,
    skipinitialspace=True
)

# Convert to numeric
df = df.apply(pd.to_numeric, errors="coerce")

print("Original shape:", df.shape)

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

print("Shape after removing empty columns:", df.shape)

# Validate structure
assert df.shape[1] == 100, (
    f"Expected 100 columns, found {df.shape[1]}"
)

print("\nDATA VALIDATION")
print("Frames:", df.shape[0])
print("Columns:", df.shape[1])
print("Joint landmarks:", df.shape[1] // 4)

print("Remaining missing values:", df.isna().sum().sum())

# Check coordinate grouping
joint_data = df.to_numpy().reshape(
    df.shape[0], 25, 4
)

print("\nReshaped joint data:", joint_data.shape)

# Inspect first joint
print("\nFirst joint, first 5 frames:")
print(joint_data[:5, 0, :])

print("\nCSV parsing successful!")
