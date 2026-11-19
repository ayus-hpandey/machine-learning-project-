
import os
import glob
import numpy as np
import pandas as pd

DATASET_PATH = r"KiMoRe"

# Find JointPosition recordings
files = glob.glob(
    os.path.join(
        DATASET_PATH,
        "**",
        "*JointPosition*.csv"
    ),
    recursive=True
)

def load_joint_data(file_path):

    df = pd.read_csv(
        file_path,
        header=None,
        skip_blank_lines=True
    )

    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Validate data structure
    if df.shape[1] != 100:
        raise ValueError(
            f"Expected 100 columns, found {df.shape[1]}"
        )

    # Frames x 25 joints x 4 values
    data = df.to_numpy().reshape(-1, 25, 4)

    return data


def extract_features(data):

    features = {}

    # Extract XYZ coordinates
    positions = data[:, :, :3]

    # Frame-to-frame displacement
    displacement = np.diff(positions, axis=0)

    # Velocity in coordinate units per frame
    velocity = np.linalg.norm(
        displacement, axis=2
    )

    # Acceleration in coordinate units per frame squared
    acceleration = np.diff(velocity, axis=0)

    # Jerk proxy
    jerk = np.diff(acceleration, axis=0)

    for joint in range(25):

        joint_name = f"joint_{joint+1}"

        xyz = positions[:, joint, :]
        joint_velocity = velocity[:, joint]

        joint_acceleration = acceleration[:, joint]
        joint_jerk = jerk[:, joint]

        # Position statistics
        for axis, axis_name in enumerate(["x", "y", "z"]):

            values = xyz[:, axis]

            features[f"{joint_name}_{axis_name}_mean"] = np.mean(values)
            features[f"{joint_name}_{axis_name}_std"] = np.std(values)
            features[f"{joint_name}_{axis_name}_range"] = np.ptp(values)

        # Movement features
        features[f"{joint_name}_total_displacement"] = np.sum(
            joint_velocity
        )

        features[f"{joint_name}_mean_velocity"] = np.mean(
            joint_velocity
        )

        features[f"{joint_name}_peak_velocity"] = np.max(
            joint_velocity
        )

        features[f"{joint_name}_mean_acceleration"] = np.mean(
            np.abs(joint_acceleration)
        )

        features[f"{joint_name}_peak_acceleration"] = np.max(
            np.abs(joint_acceleration)
        )

        features[f"{joint_name}_mean_jerk"] = np.mean(
            np.abs(joint_jerk)
        )

    features["frame_count"] = len(data)

    return features


# Test on first recording
sample_file = files[0]

data = load_joint_data(sample_file)

features = extract_features(data)

feature_df = pd.DataFrame([features])

print("Recording:", sample_file)
print("Joint data shape:", data.shape)
print("Extracted features:", feature_df.shape[1])

print("\nExtracted feature sample:")
print(feature_df.iloc[:, :15].T)

print("\nFeature extraction test successful!")
