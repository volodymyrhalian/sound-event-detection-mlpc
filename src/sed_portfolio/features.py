"""Reconstructed feature extractor. Not identical to missing university implementation.

This implementation reconstructs the described 32 ms frames, 1-second aggregation,
0.5-second hop, 128 mel bins, 32 MFCCs, and challenge's 960-dimensional vectors.
Original frame hop, filter normalization and annotation alignment are unknown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, FEATURE_GROUPS, FEATURE_NAMES, SAMPLE_RATE, STATS, WINDOW_SECONDS, WINDOW_HOP_SECONDS


def _as_2d(feature: np.ndarray) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float64)
    return arr[np.newaxis, :] if arr.ndim == 1 else arr


def feature_frames(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, np.ndarray]:
    """Compute one aligned feature vector per short-time frame; return shape (D, frames)."""
    if y.ndim != 1 or not len(y):
        raise ValueError("Expected nonempty mono audio")
    n_fft = 512  # 32 ms at 16 kHz
    hop_length = 256  # 16 ms: reconstructed assumption, not verified original
    s_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    magnitude = np.abs(s_complex)
    power_spectrum = magnitude ** 2
    mel = librosa.feature.melspectrogram(S=power_spectrum, sr=sr, n_mels=128)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel + 1e-10), n_mfcc=32)
    mfcc_d = librosa.feature.delta(mfcc, mode="nearest")
    mfcc_d2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
    norm_spec = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-12)
    flux = np.linalg.norm(np.diff(norm_spec, axis=1, prepend=norm_spec[:, :1]), axis=0)
    rms = librosa.feature.rms(S=magnitude, frame_length=n_fft)[0]
    result = {
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length),
        "melspect": mel,
        "mfcc": mfcc,
        "mfcc_d": mfcc_d,
        "mfcc_d2": mfcc_d2,
        "flux": flux,
        "flatness": librosa.feature.spectral_flatness(S=power_spectrum),
        "centroid": librosa.feature.spectral_centroid(S=magnitude, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=magnitude, sr=sr),
        "contrast": librosa.feature.spectral_contrast(S=magnitude, sr=sr),
        "rolloff_low": librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.10),
        "rolloff_high": librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.90),
        "energy": rms,
        "power": rms ** 2,
    }
    result = {k: _as_2d(v) for k, v in result.items()}
    sizes = {v.shape[1] for v in result.values()}
    if len(sizes) != 1:
        raise RuntimeError(f"Feature frames are not aligned: {sizes}")
    return result


def aggregate_frames(frames: dict[str, np.ndarray], duration: float, sr: int = SAMPLE_RATE) -> dict[str, np.ndarray]:
    """Aggregate short frames into one-second windows stepped every half second."""
    if duration <= 0:
        raise ValueError("Audio duration must be positive")
    count = max(1, int(np.floor(max(0, duration - WINDOW_SECONDS) / WINDOW_HOP_SECONDS)) + 1)
    starts = np.arange(count, dtype=np.float64) * WINDOW_HOP_SECONDS
    ends = np.minimum(starts + WINDOW_SECONDS, duration)
    first = next(iter(frames.values()))
    frame_times = librosa.frames_to_time(np.arange(first.shape[1]), sr=sr, hop_length=256)
    aggregated: dict[str, np.ndarray] = {"start_time": starts, "end_time": ends}
    for group in FEATURE_GROUPS:
        values = frames[group]
        chunks = []
        for start, end in zip(starts, ends):
            idx = np.where((frame_times >= start) & (frame_times < end))[0]
            if not len(idx):
                idx = np.array([int(np.argmin(np.abs(frame_times - start)))])
            chunks.append(values[:, idx])
        for name in STATS:
            op = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}[name]
            aggregated[f"{group}_{name}"] = np.vstack([op(segment, axis=1) for segment in chunks]).astype(np.float32)
    return aggregated


def annotator_matrix(starts: np.ndarray, ends: np.ndarray, annotation_df: pd.DataFrame,
                     filename: str, class_names: tuple[str, ...] = CLASS_NAMES) -> np.ndarray:
    """Convert interval annotations to proportional coverage by annotator (T, C, A)."""
    sub = annotation_df.loc[annotation_df["filename"].astype(str) == filename]
    annotators = sorted(sub["annotator_id"].astype(str).unique().tolist())
    if not annotators:
        raise ValueError(f"No annotations were found for {filename}; cannot create labels")
    class_idx = {name: j for j, name in enumerate(class_names)}
    annotator_idx = {name: j for j, name in enumerate(annotators)}
    output = np.zeros((len(starts), len(class_names), len(annotators)), dtype=np.float32)
    for row in sub.itertuples(index=False):
        if row.annotation not in class_idx:
            continue
        j = class_idx[row.annotation]
        k = annotator_idx[str(row.annotator_id)]
        start, end = float(row.onset), float(row.offset)
        overlap = np.maximum(0.0, np.minimum(ends, end) - np.maximum(starts, start))
        fraction = overlap / np.maximum(ends-starts, 1e-8)
        # Max merges overlapping intervals of same annotator/class without double-counting.
        output[:, j, k] = np.maximum(output[:, j, k], fraction)
    return output


def build_feature_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate stats in the feature order of the surviving challenge notebook."""
    missing = set(FEATURE_NAMES).difference(data)
    if missing:
        raise KeyError(f"Feature keys missing: {sorted(missing)}")
    arrays = [np.asarray(data[k]).reshape(len(data[k]), -1) for k in FEATURE_NAMES]
    result = np.concatenate(arrays, axis=1).astype(np.float32)
    if result.shape[1] != 960:
        raise ValueError(f"Expected 960 feature dimensions, got {result.shape[1]}")
    return result


def extract_one(path: Path, annotations: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    # librosa supports wav/flac. Compressed codecs (e.g. m4a) may require ffmpeg/audioread.
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    aggregated = aggregate_frames(feature_frames(audio, sr), len(audio)/sr, sr)
    aggregated["class_names"] = np.asarray(CLASS_NAMES)
    if annotations is not None:
        aggregated["annotations"] = annotator_matrix(
            aggregated["start_time"], aggregated["end_time"], annotations, path.name)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()
    if not args.audio_dir.is_dir():
        parser.error(f"Audio directory does not exist: {args.audio_dir}")
    annotations = None
    if args.annotations is not None:
        annotations = pd.read_csv(args.annotations, dtype={"filename": "string", "annotator_id": "string"})
        missing_cols = set(["filename", "annotator_id", "annotation", "onset", "offset"]) - set(annotations.columns)
        if missing_cols:
            parser.error(f"Annotations CSV missing columns: {sorted(missing_cols)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supported = {".wav", ".flac", ".ogg", ".m4a", ".mp3"}
    paths = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() in supported)
    if args.max_files > 0:
        paths = paths[:args.max_files]
    if not paths:
        parser.error("No supported audio files found")
    for i, path in enumerate(paths, start=1):
        data = extract_one(path, annotations)
        np.savez_compressed(args.output_dir / f"{path.stem}.npz", **data)
        print(f"[{i}/{len(paths)}] {path.name}: {len(data['start_time'])} windows")
    print(f"Done. Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
