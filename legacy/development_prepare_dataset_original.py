import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# 1. PATHS
# ============================================================

features_dir = Path("MLPC2026_dataset_development/audio_features")
output_dir = Path("prepared_outputs")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD ALL .NPZ FILES AND CREATE LABELS
# ============================================================

all_features = []
all_soft_labels = []
all_labels = []
all_filenames = []
all_start_times = []
all_end_times = []

class_names = None

for file_path in sorted(features_dir.glob("*.npz")):
    data = dict(np.load(file_path, allow_pickle=True))

    # ------------------------------------------------------------
    # Features
    # We use mfcc_mean as input features.
    # Shape: [T, 32]
    # ------------------------------------------------------------
    X = data["mfcc_mean"]

    # ------------------------------------------------------------
    # Save class names once
    # ------------------------------------------------------------
    if class_names is None:
        class_names = data["class_names"]

    # ------------------------------------------------------------
    # Annotations
    # Shape: [T, C, A]
    # T = time segments
    # C = classes
    # A = annotators
    # ------------------------------------------------------------
    annotations = data["annotations"]

    # ------------------------------------------------------------
    # Label Aggregation
    # Convert [T, C, A] -> [T, C]
    # ------------------------------------------------------------
    soft_labels = annotations.mean(axis=2)

    # Binary labels using threshold 0.5
    labels = (soft_labels >= 0.5).astype(int)

    # ------------------------------------------------------------
    # Save arrays
    # ------------------------------------------------------------
    all_features.append(X)
    all_soft_labels.append(soft_labels)
    all_labels.append(labels)

    # Save recording id for each segment
    all_filenames.extend([file_path.stem] * labels.shape[0])

    # Save segment time boundaries for case study later
    all_start_times.append(data["start_time"])
    all_end_times.append(data["end_time"])


# Combine all recordings into one dataset
X_all = np.vstack(all_features)
Y_soft = np.vstack(all_soft_labels)
Y = np.vstack(all_labels)

all_filenames = np.array(all_filenames)
all_start_times = np.concatenate(all_start_times)
all_end_times = np.concatenate(all_end_times)

print("====================================")
print("Loaded dataset")
print("====================================")
print("X shape:", X_all.shape)
print("Soft labels shape:", Y_soft.shape)
print("Binary labels shape:", Y.shape)
print("Number of segments:", Y.shape[0])
print("Number of classes:", Y.shape[1])
print("Number of recordings:", len(set(all_filenames)))
print("Class names:", class_names)
print()


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT BY RECORDING
# ============================================================

# Important:
# We split by recording/file, not by individual segments.
# This prevents segments from the same recording appearing
# in both train and test.

groups = all_filenames

# First split: 70% train, 30% temporary
gss1 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=42
)

train_idx, temp_idx = next(
    gss1.split(X_all, Y, groups=groups)
)

# Second split: temporary set into 15% validation and 15% test
temp_groups = groups[temp_idx]

gss2 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=42
)

val_relative_idx, test_relative_idx = next(
    gss2.split(X_all[temp_idx], Y[temp_idx], groups=temp_groups)
)

val_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]


# Create final datasets
X_train = X_all[train_idx]
Y_train = Y[train_idx]
Y_soft_train = Y_soft[train_idx]

X_val = X_all[val_idx]
Y_val = Y[val_idx]
Y_soft_val = Y_soft[val_idx]

X_test = X_all[test_idx]
Y_test = Y[test_idx]
Y_soft_test = Y_soft[test_idx]

filenames_train = all_filenames[train_idx]
filenames_val = all_filenames[val_idx]
filenames_test = all_filenames[test_idx]

start_times_train = all_start_times[train_idx]
start_times_val = all_start_times[val_idx]
start_times_test = all_start_times[test_idx]

end_times_train = all_end_times[train_idx]
end_times_val = all_end_times[val_idx]
end_times_test = all_end_times[test_idx]


print("====================================")
print("Split sizes")
print("====================================")
print("Train segments:", X_train.shape[0])
print("Validation segments:", X_val.shape[0])
print("Test segments:", X_test.shape[0])
print()

print("Train recordings:", len(set(filenames_train)))
print("Validation recordings:", len(set(filenames_val)))
print("Test recordings:", len(set(filenames_test)))
print()


# ============================================================
# 4. CHECK INFORMATION LEAKAGE
# ============================================================

train_files = set(filenames_train)
val_files = set(filenames_val)
test_files = set(filenames_test)

train_val_overlap = len(train_files & val_files)
train_test_overlap = len(train_files & test_files)
val_test_overlap = len(val_files & test_files)

print("====================================")
print("Leakage check")
print("====================================")
print("Train/Validation recording overlap:", train_val_overlap)
print("Train/Test recording overlap:", train_test_overlap)
print("Validation/Test recording overlap:", val_test_overlap)
print()

if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
    print("No recording-level leakage detected.")
else:
    print("Warning: recording-level leakage detected.")
print()


# ============================================================
# 5. CLASS DISTRIBUTION TABLE
# ============================================================

def class_distribution(Y_split):
    """
    Calculates class prevalence.
    For each class, this gives the fraction of segments
    where the class is active.
    """
    return Y_split.mean(axis=0)


train_dist = class_distribution(Y_train)
val_dist = class_distribution(Y_val)
test_dist = class_distribution(Y_test)
full_dist = class_distribution(Y)

distribution_table = pd.DataFrame({
    "class_index": np.arange(Y.shape[1]),
    "class_name": class_names,
    "full_dataset": full_dist,
    "train": train_dist,
    "validation": val_dist,
    "test": test_dist
})

print("====================================")
print("Class distribution")
print("====================================")
print(distribution_table)
print()

distribution_table.to_csv(output_dir / "class_distribution_splits.csv", index=False)
print("Saved:", output_dir / "class_distribution_splits.csv")
print()


# ============================================================
# 6. PREPROCESSING / STANDARDIZATION
# ============================================================

# Important:
# Fit scaler only on training data.
# Then apply the same scaler to validation and test.
# This prevents preprocessing leakage.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("====================================")
print("Preprocessing")
print("====================================")
print("X_train_scaled mean:", X_train_scaled.mean())
print("X_train_scaled std:", X_train_scaled.std())
print()


# ============================================================
# 7. SAVE PREPARED DATASET FOR NEXT TASKS
# ============================================================

np.savez_compressed(
    output_dir / "prepared_dataset.npz",

    # Raw features
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,

    # Scaled features
    X_train_scaled=X_train_scaled,
    X_val_scaled=X_val_scaled,
    X_test_scaled=X_test_scaled,

    # Binary labels
    Y_train=Y_train,
    Y_val=Y_val,
    Y_test=Y_test,

    # Soft labels before threshold
    Y_soft_train=Y_soft_train,
    Y_soft_val=Y_soft_val,
    Y_soft_test=Y_soft_test,

    # Metadata for segments
    filenames_train=filenames_train,
    filenames_val=filenames_val,
    filenames_test=filenames_test,

    start_times_train=start_times_train,
    start_times_val=start_times_val,
    start_times_test=start_times_test,

    end_times_train=end_times_train,
    end_times_val=end_times_val,
    end_times_test=end_times_test,

    # Class names
    class_names=class_names
)

print("Saved:", output_dir / "prepared_dataset.npz")


# ============================================================
# 8. SAVE SCALER
# ============================================================

joblib.dump(scaler, output_dir / "standard_scaler.joblib")
print("Saved:", output_dir / "standard_scaler.joblib")


# ============================================================
# 9. SAVE SPLIT INDICES
# ============================================================

np.savez_compressed(
    output_dir / "data_splits.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

print("Saved:", output_dir / "data_splits.npz")


# ============================================================
# 10. SAVE SPLIT SUMMARY
# ============================================================

summary = {
    "feature_used": "mfcc_mean",
    "label_aggregation": "mean over annotators + threshold 0.5",
    "split_strategy": "recording-level split",
    "split_ratio": "approximately 70/15/15",
    "num_segments_total": int(Y.shape[0]),
    "num_classes": int(Y.shape[1]),
    "num_recordings_total": int(len(set(all_filenames))),
    "train_segments": int(X_train.shape[0]),
    "validation_segments": int(X_val.shape[0]),
    "test_segments": int(X_test.shape[0]),
    "train_recordings": int(len(set(filenames_train))),
    "validation_recordings": int(len(set(filenames_val))),
    "test_recordings": int(len(set(filenames_test))),
    "train_validation_recording_overlap": int(train_val_overlap),
    "train_test_recording_overlap": int(train_test_overlap),
    "validation_test_recording_overlap": int(val_test_overlap),
    "preprocessing": "StandardScaler fitted only on training data"
}

with open(output_dir / "split_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("Saved:", output_dir / "split_summary.json")