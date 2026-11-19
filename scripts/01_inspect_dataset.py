
import os
import glob
import pandas as pd
import numpy as np

# Original KIMORE dataset
DATASET_PATH = r"KiMoRe"

print("=" * 60)
print("KIMORE DATASET INSPECTION")
print("=" * 60)

# Verify directory
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("KIMORE dataset directory not found!")

print("\nDataset directory exists!")

# List top-level folders
print("\nTop-level contents:")
for item in os.listdir(DATASET_PATH):
    print("-", item)

# Find JointPosition files
csv_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*JointPosition*.csv"),
    recursive=True
)

print("\nTotal JointPosition CSV files:", len(csv_files))

# Inspect first 3 files
for file in csv_files[:3]:

    print("\n" + "=" * 60)
    print("FILE:", file)

    try:
        df = pd.read_csv(file, sep=None, engine="python")

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())

    except Exception as e:
        print("Error:", e)
