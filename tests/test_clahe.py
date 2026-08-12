import numpy as np
from src.clahe import clip_histogram

def test_clip_histogram_no_clip():
    hist = np.array([1, 2, 3, 4])
    out = clip_histogram(hist, 10)
    assert np.array_equal(out, hist)

def test_clip_histogram_preserves_sum():
    hist = np.array([10, 0, 0, 0])
    out = clip_histogram(hist, 4)
    assert out.sum() == hist.sum()

def test_clip_histogram_remainder():
    hist = np.array([9, 0, 0, 0])
    out = clip_histogram(hist, 4)
    assert out.sum() == hist.sum()
    assert len(out) == len(hist)

def test_clip_histogram_empty():
    hist = np.array([])
    out = clip_histogram(hist, 4)
    assert out.size == 0