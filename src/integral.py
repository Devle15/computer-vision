import numpy as np


def integral_image(img):
    """Compute a padded summed-area table."""
    img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError("img must be a 2D array")

    img64 = img.astype(np.int64)

    H, W = img64.shape

    S = np.zeros(
        (H + 1, W + 1),
        dtype=np.int64,
    )

    S[1:, 1:] = np.cumsum(
        np.cumsum(img64, axis=0),
        axis=1,
    )

    return S


def box_sum(S, y1, x1, y2, x2):
    """Sum inclusive rectangle using an integral image."""
    S = np.asarray(S)

    return (
        S[np.asarray(y2) + 1, np.asarray(x2) + 1]
        - S[np.asarray(y1), np.asarray(x2) + 1]
        - S[np.asarray(y2) + 1, np.asarray(x1)]
        + S[np.asarray(y1), np.asarray(x1)]
    )


def local_mean_std(img, radius):
    """Return local mean and standard deviation."""
    img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError("img must be a 2D array")

    if radius < 0:
        raise ValueError("radius must be non-negative")

    H, W = img.shape

    y = np.arange(H)
    x = np.arange(W)

    y1 = np.maximum(y[:, None] - radius, 0)
    x1 = np.maximum(x[None, :] - radius, 0)

    y2 = np.minimum(y[:, None] + radius, H - 1)
    x2 = np.minimum(x[None, :] + radius, W - 1)

    S = integral_image(img)
    S2 = integral_image(
        img.astype(np.int64) ** 2
    )

    sums = box_sum(
        S,
        y1,
        x1,
        y2,
        x2,
    )

    sums2 = box_sum(
        S2,
        y1,
        x1,
        y2,
        x2,
    )

    area = (
        (y2 - y1 + 1)
        * (x2 - x1 + 1)
    ).astype(np.float64)

    mean = sums.astype(
        np.float64
    ) / area

    variance = (
        sums2.astype(np.float64) / area
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        0.0,
    )

    std = np.sqrt(variance)

    return mean, std


def adaptive_threshold(
    img,
    radius=15,
    method="bradley",
    t=0.15,
    k=0.34,
):
    """Adaptive threshold using Bradley-Roth or Sauvola."""
    img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError("img must be a 2D array")

    if radius < 0:
        raise ValueError("radius must be non-negative")

    method = method.lower()

    mean, std = local_mean_std(
        img,
        radius,
    )

    if method == "bradley":
        threshold = mean * (1.0 - t)

    elif method == "sauvola":
        R = 128.0

        threshold = mean * (
            1.0
            + k * (std / R - 1.0)
        )

    else:
        raise ValueError(
            "method must be 'bradley' or 'sauvola'"
        )

    result = np.where(
        img > threshold,
        255,
        0,
    ).astype(np.uint8)

    return result