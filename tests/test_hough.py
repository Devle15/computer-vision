import numpy as np

from src.hough import convolve2d, find_peaks, hough_accumulate, hough_multiscale


def test_convolve2d_matches_expected():
    img = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], dtype=float)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=float) / 9
    out = convolve2d(img, kernel)
    assert out.shape == img.shape
    assert np.allclose(out[1, 1], 1.0 / 9.0)


def test_hough_accumulate_returns_grid_shape():
    edges = np.zeros((10, 10), dtype=float)
    edges[5, 5] = 1.0
    acc = hough_accumulate(edges, theta_range=(-90, 90), rho_range=(-20, 20), n_theta=180, n_rho=41)
    assert acc.shape == (41, 180)
    assert acc.sum() > 0


def test_find_peaks_returns_highest_local_maxima():
    A = np.zeros((9, 9), dtype=float)
    A[4, 4] = 10.0
    A[4, 3] = 8.0
    A[3, 4] = 7.0
    A[5, 4] = 6.0
    A[4, 5] = 5.0
    peaks = find_peaks(A, num_peaks=3, nms_radius=1, smooth_sigma=0.0)
    assert peaks.shape[1] == 2
    assert (4, 4) in tuple(map(tuple, peaks))


def test_hough_multiscale_returns_valid_peak_coords():
    edges = np.zeros((16, 16), dtype=float)
    for y in range(4, 12):
        edges[y, 8] = 1.0
    peaks = hough_multiscale(edges, n_peaks=4, n_levels=2, refine_factor=4, n_theta0=90, n_rho0=32)
    assert isinstance(peaks, np.ndarray)
    assert peaks.ndim == 2
    assert peaks.shape[1] == 2
    assert peaks.shape[0] <= 4
