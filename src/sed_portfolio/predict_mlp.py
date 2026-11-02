"""Inference using original or newly trained multi-label MLP weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .features import build_feature_matrix
from .models import mlp_from_preset, PRESETS
from .constants import CLASS_NAMES
from .temporal import labels_to_intervals


def predict_features(feature_paths: list[Path], checkpoint: Path, preset: str, threshold: float = 0.5) -> pd.DataFrame:
    model = mlp_from_preset(preset, output_dim=len(CLASS_NAMES))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    dfs = []
    with torch.no_grad():
        for path in feature_paths:
            with np.load(path, allow_pickle=False) as data:
                x = data["mfcc_mean"].astype(np.float32) if preset == "development" else build_feature_matrix(data)
                starts = data["start_time"]
            if x.shape[1] != PRESETS[preset]["input_dim"]:
                raise ValueError(f"Unexpected feature dimension in {path}: {x.shape}")
            preds = []
            for chunk in np.array_split(x, max(1, int(np.ceil(len(x)/512)))):
                if len(chunk):
                    probs = torch.sigmoid(model(torch.from_numpy(chunk))).numpy()
                    preds.append(probs >= threshold)
            if not preds:
                continue
            # Match the challenge: only evaluate one-second-aligned segments.
            preds_all = np.vstack(preds)
            whole_second = np.isclose(starts % 1.0, 0.0, atol=1e-6)
            dfs.append(labels_to_intervals(preds_all[whole_second], starts[whole_second],
                                            f"{path.stem}.wav", CLASS_NAMES, segment_seconds=1.0))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["filename","annotation","onset","offset"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="challenge")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()
    if not (0 < args.threshold < 1):
        parser.error("--threshold must be in (0,1)")
    paths = sorted(args.features_dir.glob("*.npz"))
    if not paths:
        parser.error(f"No feature .npz files found under {args.features_dir}")
    predictions = predict_features(paths, args.checkpoint, args.preset, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} event intervals to {args.output}")


if __name__ == "__main__":
    main()
