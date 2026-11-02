import numpy as np
import pandas as pd
import pytest
from sed_portfolio.temporal import labels_to_intervals, smooth_intervals


def test_intervals_merge_adjacent_windows():
    y = np.array([[1, 0], [1, 1], [0, 1], [1, 0]])
    start = np.array([0., 1., 2., 4.])
    got = labels_to_intervals(y, start, "sound.wav", ("a", "b"))
    assert got[["annotation","onset","offset"]].values.tolist() == [
        ["a", 0., 2.], ["a", 4., 5.], ["b", 1., 3.]]


def test_smoothing():
    events = pd.DataFrame({"filename": ["x.wav"]*3, "annotation": ["a"]*3,
                           "onset": [0., 2., 5.], "offset": [1., 3., 6.]})
    got = smooth_intervals(events, 1.0)
    assert list(got.onset) == [0.0, 5.0]
    assert list(got.offset) == [3.0, 6.0]


def test_smoothing_negative_gap_rejected():
    with pytest.raises(ValueError):
        smooth_intervals(pd.DataFrame(), -0.5)
