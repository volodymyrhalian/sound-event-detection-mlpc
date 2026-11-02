"""Translate framewise labels into event annotations and smooth event intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .constants import CLASS_NAMES

REQUIRED_COLUMNS = ["filename", "annotation", "onset", "offset"]


def labels_to_intervals(predictions: np.ndarray, start_times: np.ndarray,
                        filename: str, class_names=CLASS_NAMES, segment_seconds: float = 1.0) -> pd.DataFrame:
    preds = np.asarray(predictions)
    starts = np.asarray(start_times, dtype=float)
    if preds.ndim != 2 or preds.shape != (len(starts), len(class_names)):
        raise ValueError("Predictions must be (T, number of classes)")
    if len(starts) and np.any(np.diff(starts) <= 0):
        raise ValueError("Start times must be strictly increasing")
    rows = []
    for c, name in enumerate(class_names):
        indices = np.flatnonzero(preds[:, c])
        if not len(indices):
            continue
        current_start = float(starts[indices[0]])
        current_end = float(starts[indices[0]] + segment_seconds)
        for idx in indices[1:]:
            s = float(starts[idx])
            e = s + segment_seconds
            if s <= current_end + 1e-8:
                current_end = max(current_end, e)
            else:
                rows.append((filename, name, current_start, current_end))
                current_start, current_end = s, e
        rows.append((filename, name, current_start, current_end))
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def smooth_intervals(events: pd.DataFrame, max_gap_seconds: float = 1.0) -> pd.DataFrame:
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be nonnegative")
    if events.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    if set(REQUIRED_COLUMNS)-set(events.columns):
        raise ValueError("Missing required event columns")
    merged = []
    for (filename, cls), group in events.groupby(["filename", "annotation"], sort=True):
        group = group.sort_values("onset")
        onset, offset = None, None
        for record in group.itertuples(index=False):
            start, end = float(record.onset), float(record.offset)
            if end < start:
                raise ValueError("End time cannot precede onset")
            if onset is None:
                onset, offset = start, end
            elif start-offset <= max_gap_seconds + 1e-9:
                offset = max(offset, end)
            else:
                merged.append((filename, cls, onset, offset))
                onset, offset = start, end
        merged.append((filename, cls, onset, offset))
    return pd.DataFrame(merged, columns=REQUIRED_COLUMNS)
