import numpy as np
import pandas as pd
from sed_portfolio.constants import CLASS_NAMES
from sed_portfolio.features import annotator_matrix, aggregate_frames, build_feature_matrix, feature_frames


def test_reconstructed_features_960_dimensions():
    sr = 16000
    y = np.sin(2*np.pi*440*np.arange(sr*2)/sr).astype(np.float32)
    data = aggregate_frames(feature_frames(y), len(y)/sr)
    x = build_feature_matrix(data)
    assert x.shape == (3, 960)
    assert np.isfinite(x).all()
    assert data["mfcc_mean"].shape == (3, 32)
    assert data["melspect_mean"].shape == (3, 128)


def test_annotator_coverage():
    ann = pd.DataFrame({"filename": ["test.wav", "test.wav"],
        "annotator_id": ["alpha", "beta"], "annotation": ["bell_ringing", "bell_ringing"],
        "onset": [0., 0.5], "offset": [1., 1.]})
    y = annotator_matrix(np.array([0.0, 0.5]), np.array([1., 1.5]), ann, "test.wav")
    assert y.shape == (2, len(CLASS_NAMES), 2)
    idx = CLASS_NAMES.index("bell_ringing")
    assert y[0, idx, 0] == 1
    assert y[0, idx, 1] == .5
    assert y[1, idx, 0] == .5
