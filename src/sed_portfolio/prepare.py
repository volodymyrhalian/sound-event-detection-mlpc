"""Rebuild prepared MLPC arrays from per-recording feature files (reconstructed)."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from .constants import CLASS_NAMES


def prepare(feature_paths: list[Path], output_path: Path) -> dict[str, np.ndarray]:
    """Recording-disjoint split 70/15/15, majority vote and train-only scaling."""
    if len(feature_paths) < 7:
        raise ValueError("Use at least seven recordings to obtain meaningful splits")
    xs, ys, y_soft, names, start_times, end_times = [], [], [], [], [], []
    for path in sorted(feature_paths):
        with np.load(path, allow_pickle=False) as data:
            x = np.asarray(data["mfcc_mean"], dtype=np.float32)
            ann = np.asarray(data["annotations"], dtype=np.float32)
            if ann.ndim != 3 or ann.shape[:2] != (len(x), len(CLASS_NAMES)):
                raise ValueError(f"Bad label dimensions: {path}")
            labels = np.nanmean(ann, axis=2)
            # If a recording has no annotator labels in the reconstructed data,
            # that recording cannot be considered a labeled training sample.
            xs.append(x)
            y_soft.append(labels)
            ys.append((labels >= 0.5).astype(np.int64))
            names.extend([path.stem] * len(x))
            start_times.append(data["start_time"])
            end_times.append(data["end_time"])
    x_all, y_all, soft_all = np.vstack(xs), np.vstack(ys), np.vstack(y_soft)
    groups = np.asarray(names)
    if x_all.shape[0] < 10:
        raise ValueError("Too few segments to create training splits")
    first_split = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train, rest = next(first_split.split(x_all, y_all, groups=groups))
    second_split = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_rel, test_rel = next(second_split.split(x_all[rest], y_all[rest], groups=groups[rest]))
    indices = {"train": train, "val": rest[val_rel], "test": rest[test_rel]}
    parts: dict[str, np.ndarray] = {"class_names": np.asarray(CLASS_NAMES)}
    start = np.concatenate(start_times)
    end = np.concatenate(end_times)
    scaler = StandardScaler().fit(x_all[train])
    for split, idx in indices.items():
        parts[f"X_{split}"] = x_all[idx]
        parts[f"Y_{split}"] = y_all[idx]
        parts[f"Y_soft_{split}"] = soft_all[idx]
        parts[f"X_{split}_scaled"] = scaler.transform(x_all[idx]).astype(np.float32)
        parts[f"filenames_{split}"] = groups[idx]
        parts[f"start_times_{split}"] = start[idx]
        parts[f"end_times_{split}"] = end[idx]
    splits = [set(groups[idx]) for idx in indices.values()]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i+1, 3)):
        raise RuntimeError("Recording split leakage detected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **parts)
    joblib.dump(scaler, output_path.with_suffix(".scaler.joblib"))
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.features_dir.glob("*.npz"))
    if not paths:
        parser.error("No NPZ feature files found")
    result = prepare(paths, args.output)
    print(f"Saved {args.output}; train {len(result['X_train'])}, val {len(result['X_val'])}, test {len(result['X_test'])} segments")


if __name__ == "__main__":
    main()
