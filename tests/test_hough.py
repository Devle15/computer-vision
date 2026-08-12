import numpy as np

from src.hough import (
    convolve2d,
    find_peaks,
    hough_accumulate,
    hough_multiscale,
)


def test_convolve2d_matches_expected():
    img = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    ) / 9

    out = convolve2d(img, kernel)

    assert out.shape == img.shape
    assert np.allclose(out[1, 1], 1.0 / 9.0)


def test_hough_accumulate_returns_grid_shape():
    edges = np.zeros((10, 10), dtype=float)
    edges[5, 5] = 1.0

    acc, thetas, rhos = hough_accumulate(
        edges,
        theta_range=(-90, 90),
        rho_range=(-20, 20),
        n_theta=180,
        n_rho=41,
    )

    assert acc.shape == (41, 180)
    assert thetas.shape == (180,)
    assert rhos.shape == (41,)
    assert acc.sum() > 0


def test_find_peaks_returns_highest_local_maxima():
    A = np.zeros((9, 9), dtype=float)

    A[4, 4] = 10.0
    A[4, 3] = 8.0
    A[3, 4] = 7.0
    A[5, 4] = 6.0
    A[4, 5] = 5.0

    peaks = find_peaks(
        A,
        num_peaks=3,
        nms_radius=1,
        smooth_sigma=0.0,
    )

    assert peaks.shape == (1, 3)
    assert np.allclose(peaks[0, :2], [4.0, 4.0])
    assert np.isclose(peaks[0, 2], 10.0)


def test_find_peaks_returns_multiple_peaks_without_nms():
    A = np.zeros((9, 9), dtype=float)

    A[1, 1] = 5.0
    A[3, 6] = 8.0
    A[7, 2] = 6.0

    peaks = find_peaks(
        A,
        num_peaks=3,
        nms_radius=0,
        smooth_sigma=0.0,
    )

    assert peaks.shape == (3, 3)

    values = sorted(
        peaks[:, 2],
        reverse=True,
    )

    assert values == [8.0, 6.0, 5.0]


def test_hough_multiscale_returns_valid_peak_coords():
    edges = np.zeros((16, 16), dtype=float)

    for y in range(4, 12):
        edges[y, 8] = 1.0

    peaks = hough_multiscale(
        edges,
        n_peaks=4,
        n_levels=2,
        refine_factor=4,
        n_theta0=90,
        n_rho0=32,
    )

    assert isinstance(peaks, np.ndarray)
    assert peaks.ndim == 2
    assert peaks.shape[1] == 2
    assert peaks.shape[0] <= 4


def test_hough_vertical_line_wraparound():
    edges = np.zeros((60, 60), dtype=float)

    # Vertical line near theta = 0 / 180 degrees.
    for y in range(5, 55):
        edges[y, 30] = 1.0

    peaks = hough_multiscale(
        edges,
        n_peaks=5,
        n_levels=2,
        refine_factor=4,
        n_theta0=180,
        n_rho0=121,
    )

    assert peaks.ndim == 2
    assert peaks.shape[1] == 2
    assert peaks.shape[0] > 0

    theta_deg = np.rad2deg(peaks[:, 1])

    # A vertical line should appear near 0 or 180 degrees.
    assert np.any(
        (theta_deg < 5.0) | (theta_deg > 175.0)
    )


def test_hough_known_vertical_line():
    edges = np.zeros((80, 80), dtype=float)

    x0 = 25

    for y in range(10, 70):
        edges[y, x0] = 1.0

    peaks = hough_multiscale(
        edges,
        n_peaks=5,
        n_levels=2,
        refine_factor=4,
        n_theta0=180,
        n_rho0=161,
    )

    assert peaks.shape[0] > 0

    rho_pred = peaks[0, 0]
    theta_pred = peaks[0, 1]

    theta_deg = np.rad2deg(theta_pred)

    # For a vertical line: rho ~= x0 and theta ~= 0 or 180 degrees.
    assert x0 - 3 <= abs(rho_pred) <= x0 + 3
    assert theta_deg < 5.0 or theta_deg > 175.0


def test_hough_long_and_short_lines():
    edges = np.zeros((100, 100), dtype=float)

    # Long vertical line.
    for y in range(10, 90):
        edges[y, 20] = 1.0

    # Short diagonal line.
    for i in range(12):
        y = 55 + i
        x = 60 + i
        edges[y, x] = 1.0

    peaks = hough_multiscale(
        edges,
        n_peaks=6,
        n_levels=2,
        refine_factor=4,
        n_theta0=180,
        n_rho0=201,
    )

    assert peaks.ndim == 2
    assert peaks.shape[1] == 2
    assert peaks.shape[0] > 0