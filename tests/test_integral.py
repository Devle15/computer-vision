import numpy as np

from src.integral import (
    integral_image,
    box_sum,
    local_mean_std,
    adaptive_threshold,
)


def test_integral_image_shape_dtype_and_values():
    img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    S = integral_image(img)

    expected = np.array([
        [0, 0, 0, 0],
        [0, 1, 3, 6],
        [0, 5, 12, 21],
        [0, 12, 27, 45],
    ], dtype=np.int64)

    assert S.shape == (4, 4)
    assert S.dtype == np.int64
    assert np.array_equal(S, expected)


def test_box_sum_scalar():
    img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    S = integral_image(img)

    assert box_sum(S, 1, 1, 1, 1) == 5
    assert box_sum(S, 1, 1, 2, 2) == 28
    assert box_sum(S, 0, 0, 2, 2) == 45


def test_box_sum_vectorized():
    img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    S = integral_image(img)

    y1 = np.array([0, 0, 1])
    x1 = np.array([0, 1, 1])
    y2 = np.array([1, 2, 2])
    x2 = np.array([1, 2, 2])

    result = box_sum(S, y1, x1, y2, x2)

    expected = np.array([12, 33, 28])

    assert np.array_equal(result, expected)


def test_box_sum_matches_direct_sum():
    img = np.arange(1, 26).reshape(5, 5)

    S = integral_image(img)

    regions = [
        (0, 0, 0, 0),
        (0, 0, 4, 4),
        (1, 1, 3, 3),
        (0, 2, 2, 4),
        (2, 0, 4, 2),
    ]

    for y1, x1, y2, x2 in regions:
        integral_result = box_sum(
            S,
            y1,
            x1,
            y2,
            x2,
        )

        direct_result = np.sum(
            img[y1:y2 + 1, x1:x2 + 1],
            dtype=np.int64,
        )

        assert integral_result == direct_result


def test_local_mean_std_constant_image():
    img = np.full(
        (5, 5),
        100,
        dtype=np.uint8,
    )

    mean, std = local_mean_std(
        img,
        radius=1,
    )

    assert mean.shape == img.shape
    assert std.shape == img.shape
    assert np.allclose(mean, 100.0)
    assert np.allclose(std, 0.0)


def test_adaptive_threshold_returns_binary_uint8():
    img = np.array([
        [0, 0, 0, 255, 255],
        [0, 0, 0, 255, 255],
        [0, 0, 128, 255, 255],
        [0, 0, 0, 255, 255],
        [0, 0, 0, 255, 255],
    ], dtype=np.uint8)

    for method in ("bradley", "sauvola"):
        result = adaptive_threshold(
            img,
            radius=1,
            method=method,
        )

        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert np.all(
            np.isin(result, [0, 255])
        )