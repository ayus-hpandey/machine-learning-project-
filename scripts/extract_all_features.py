
import os
import glob
import numpy as np
import pandas as pd

from pathlib import Path

# Dataset location
DATASET_PATH = Path("KiMoRe")

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Find all recordings
files = glob.glob(
    str(DATASET_PATH / "**" / "*JointPosition*.csv"),
    recursive=True
)

print("Total recordings found:", len(files))


def load_joint_data(file_path):

    df = pd.read_csv(
        file_path,
        header=None,
        skip_blank_lines=True
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != 100:
        raise ValueError(
            f"Expected 100 columns, got {df.shape[1]}"
        )

    if df.isna().any().any():
        raise ValueError("Missing coordinate values detected")

    data = df.to_numpy().reshape(-1, 25, 4)

    return data


def extract_features(data):

    features = {}

    positions = data[:, :, :3]

    displacement = np.diff(positions, axis=0)

    velocity = np.linalg.norm(
        displacement, axis=2
    )

    acceleration = np.diff(velocity, axis=0)

    jerk = np.diff(acceleration, axis=0)

    for joint in range(25):

        name = f"joint_{joint+1}"

        xyz = positions[:, joint, :]
        v = velocity[:, joint]
        a = acceleration[:, joint]
        j = jerk[:, joint]

        # Position statistics
        for axis, axis_name in enumerate(["x", "y", "z"]):

            values = xyz[:, axis]

            features[f"{name}_{axis_name}_mean"] = np.mean(values)
            features[f"{name}_{axis_name}_std"] = np.std(values)
            features[f"{name}_{axis_name}_range"] = np.ptp(values)

        # Movement statistics
        features[f"{name}_total_displacement"] = np.sum(v)
        features[f"{name}_mean_velocity"] = np.mean(v)
        features[f"{name}_peak_velocity"] = np.max(v)

        features[f"{name}_mean_acceleration"] = np.mean(np.abs(a))
        features[f"{name}_peak_acceleration"] = np.max(np.abs(a))

        features[f"{name}_mean_jerk"] = np.mean(np.abs(j))

    features["frame_count"] = len(data)

    return features


# Process all recordings
all_records = []
failed_files = []

for i, file_path in enumerate(files):

    try:

        data = load_joint_data(file_path)

        features = extract_features(data)

        # Extract metadata from directory structure
        relative_path = Path(file_path).relative_to(DATASET_PATH)
        parts = relative_path.parts

        if len(parts) < 5:
            raise ValueError("Unexpected directory structure")

        dataset_group = parts[0]
        condition = parts[1]
        patient_id = parts[2]
        exercise_id = parts[3]

        record = {
            "recording_id": Path(file_path).stem,
            "dataset_group": dataset_group,
            "condition": condition,
            "patient_id": patient_id,
            "exercise_id": exercise_id,
            **features
        }

        all_records.append(record)

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(files)} recordings")

    except Exception as e:

        failed_files.append({
            "file": file_path,
            "error": str(e)
        })

        print("Failed:", file_path, "|", e)


# Create feature dataset
feature_df = pd.DataFrame(all_records)

# Save results
output_path = OUTPUT_DIR / "kimore_extracted_features.csv"

feature_df.to_csv(output_path, index=False)

# Save failed files separately
pd.DataFrame(failed_files).to_csv(
    OUTPUT_DIR / "failed_recordings.csv",
    index=False
)

# Summary
print("\n" + "=" * 55)
print("KIMORE FEATURE EXTRACTION SUMMARY")
print("=" * 55)

print("Total files found:", len(files))
print("Successfully processed:", len(all_records))
print("Failed recordings:", len(failed_files))

print("Final dataset shape:", feature_df.shape)
print("Features per recording:", 376)

print("\nCondition distribution:")
print(feature_df["condition"].value_counts())

print("\nUnique patients:", feature_df["patient_id"].nunique())

print("\nMissing values:", feature_df.isna().sum().sum())

print("\nSaved:", output_path)
