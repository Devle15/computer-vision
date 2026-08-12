import numpy as np


def convolve2d(img, kernel):
    """Simple 2D convolution using full padding and a flipped kernel."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2D array")

    ky, kx = kernel.shape
    if ky == 0 or kx == 0:
        raise ValueError("kernel cannot be empty")

    pad_y = ky // 2
    pad_x = kx // 2
    padded = np.pad(
        img,
        ((pad_y, ky - pad_y - 1 if ky % 2 == 0 else pad_y),
         (pad_x, kx - pad_x - 1 if kx % 2 == 0 else pad_x)),
        mode="edge"
    )

    flipped = kernel[::-1, ::-1]
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded[i:i + ky, j:j + kx]
            out[i, j] = np.sum(patch * flipped)
    return out


def _as_theta_range(theta_range):
    if theta_range is None:
        return (0.0, 180.0)
    if isinstance(theta_range, (tuple, list, np.ndarray)) and len(theta_range) == 2:
        return float(theta_range[0]), float(theta_range[1])
    raise ValueError("theta_range must be a 2-item tuple/list")


def _as_rho_range(rho_range, edges):
    diag = np.hypot(edges.shape[0], edges.shape[1])
    if rho_range is None:
        return (-diag, diag)
    if isinstance(rho_range, (tuple, list, np.ndarray)) and len(rho_range) == 2:
        return float(rho_range[0]), float(rho_range[1])
    raise ValueError("rho_range must be a 2-item tuple/list")


def _normalize_theta(theta):
    return np.mod(theta, np.pi)


def _angle_diff(theta_a, theta_b):
    diff = np.abs(np.mod(theta_a - theta_b + np.pi / 2.0, np.pi) - np.pi / 2.0)
    return diff


def _gaussian_kernel(sigma):
    radius = max(1, int(np.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel


def _smooth_accumulator(A, sigma):
    if sigma is None or sigma <= 0:
        return A.copy()
    kernel = _gaussian_kernel(float(sigma))
    return convolve2d(A, kernel)


def _local_maxima(filtered, radius):
    h, w = filtered.shape
    peak_mask = np.zeros_like(filtered, dtype=bool)
    for i in range(h):
        i0 = max(0, i - radius)
        i1 = min(h, i + radius + 1)
        for j in range(w):
            j0 = max(0, j - radius)
            j1 = min(w, j + radius + 1)
            patch = filtered[i0:i1, j0:j1]
            if filtered[i, j] == np.max(patch) and filtered[i, j] > 0:
                peak_mask[i, j] = True
    return peak_mask


def _theta_windows(theta_center, delta_theta):
    low = theta_center - delta_theta
    high = theta_center + delta_theta
    windows = []
    if low >= 0.0 and high <= np.pi:
        windows.append((low, high, 1.0))
    else:
        if low < 0.0:
            windows.append((0.0, high, 1.0))
            windows.append((np.pi + low, np.pi, -1.0))
        if high > np.pi:
            windows.append((low, np.pi, 1.0))
            windows.append((0.0, high - np.pi, -1.0))
    return windows


def hough_accumulate(
    edges,
    theta_range,
    rho_range,
    n_theta=180,
    n_rho=None,
    grad_dir=None,
    delta_deg=None,
):
    """Compute a line Hough accumulator.

    Returns:
        A      : accumulator of shape (n_rho, n_theta)
        thetas : theta values in radians, normalized to [0, pi)
        rhos   : rho values corresponding to accumulator rows
    """
    edges = np.asarray(edges, dtype=np.float64)

    if edges.ndim != 2:
        raise ValueError("edges must be a 2D array")

    if n_theta <= 0:
        raise ValueError("n_theta must be positive")

    theta_min_deg, theta_max_deg = _as_theta_range(theta_range)
    rho_min, rho_max = _as_rho_range(rho_range, edges)

    if n_rho is None:
        n_rho = int(np.ceil(np.hypot(edges.shape[0], edges.shape[1]))) + 1

    n_rho = int(n_rho)

    if n_rho <= 0:
        raise ValueError("n_rho must be positive")

    # Hough convention: theta in [0, pi)
    theta_min = np.deg2rad(theta_min_deg)
    theta_max = np.deg2rad(theta_max_deg)

    if theta_max <= theta_min:
        theta_max += np.pi

    thetas = np.linspace(
        theta_min,
        theta_max,
        n_theta,
        endpoint=False,
        dtype=np.float64,
    )

    thetas = np.mod(thetas, np.pi)

    rhos = np.linspace(
        rho_min,
        rho_max,
        n_rho,
        endpoint=True,
        dtype=np.float64,
    )

    if n_rho > 1:
        drho = (rho_max - rho_min) / (n_rho - 1)
    else:
        drho = 1.0

    acc = np.zeros((n_rho, n_theta), dtype=np.float64)

    # Gradient direction
    if grad_dir is not None:
        grad_dir_arr = np.asarray(grad_dir, dtype=np.float64)

        if grad_dir_arr.shape != edges.shape:
            raise ValueError(
                "grad_dir must have the same shape as edges"
            )
    else:
        grad_dir_arr = None

    y_idx, x_idx = np.nonzero(edges > 0)

    if y_idx.size == 0:
        return acc, thetas, rhos

    for y, x in zip(y_idx, x_idx):

        # --------------------------------------------------
        # Determine theta candidates
        # --------------------------------------------------
        if grad_dir_arr is not None:

            grad_theta = _normalize_theta(
                np.deg2rad(float(grad_dir_arr[y, x]))
            )

            if delta_deg is not None:

                delta = np.deg2rad(float(delta_deg))

                valid = (
                    _angle_diff(thetas, grad_theta)
                    <= delta
                )

                theta_candidates = thetas[valid]

                if theta_candidates.size == 0:
                    continue

            else:
                theta_candidates = thetas

        else:
            theta_candidates = thetas

        # --------------------------------------------------
        # Voting
        # --------------------------------------------------
        for theta in theta_candidates:

            rho = (
                x * np.cos(theta)
                + y * np.sin(theta)
            )

            rho_bin = int(
                np.clip(
                    np.round(
                        (rho - rho_min) / drho
                    ),
                    0,
                    n_rho - 1,
                )
            )

            theta_bin = int(
                np.argmin(
                    _angle_diff(thetas, theta)
                )
            )

            acc[rho_bin, theta_bin] += 1.0

    return acc, thetas, rhos


def find_peaks(
    A,
    thetas=None,
    rhos=None,
    num_peaks=8,
    nms_radius=6,
    smooth_sigma=1.0,
):
    """Return strongest Hough peaks as (rho, theta, votes)."""
    A = np.asarray(A, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if thetas is None:
        thetas = np.arange(
            A.shape[1],
            dtype=np.float64,
        )
    else:
        thetas = np.asarray(
            thetas,
            dtype=np.float64,
        )

    if rhos is None:
        rhos = np.arange(
            A.shape[0],
            dtype=np.float64,
        )
    else:
        rhos = np.asarray(
            rhos,
            dtype=np.float64,
        )

    if len(thetas) != A.shape[1]:
        raise ValueError(
            "thetas length must match A.shape[1]"
        )

    if len(rhos) != A.shape[0]:
        raise ValueError(
            "rhos length must match A.shape[0]"
        )

    if num_peaks <= 0:
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    # Smooth ONLY for locating peaks.
    filtered = _smooth_accumulator(
        A,
        float(smooth_sigma),
    ) if smooth_sigma is not None else A.copy()

    radius = max(
        0,
        int(nms_radius),
    )

    local_max = _local_maxima(
        filtered,
        radius,
    )

    if not np.any(local_max):
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    ys, xs = np.where(local_max)

    # Rank by smoothed response
    values = filtered[ys, xs]
    order = np.argsort(
        values
    )[::-1]

    selected = []

    for idx in order:

        y = int(ys[idx])
        x = int(xs[idx])

        # NMS
        too_close = False

        if radius > 0:
            for sy, sx in selected:
                if (
                    abs(y - sy) <= radius
                    and abs(x - sx) <= radius
                ):
                    too_close = True
                    break

        if too_close:
            continue

        selected.append(
            (y, x)
        )

        if len(selected) >= num_peaks:
            break

    peaks = []

    for y, x in selected:

        rho = float(rhos[y])
        theta = float(thetas[x])

        # IMPORTANT:
        # votes come from original accumulator,
        # not the smoothed accumulator.
        votes = float(A[y, x])

        peaks.append(
            (rho, theta, votes)
        )

    if not peaks:
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    return np.asarray(
        peaks,
        dtype=np.float64,
    )


def hough_multiscale(
    edges,
    grad_dir=None,
    n_peaks=8,
    n_levels=2,
    refine_factor=8,
    n_theta0=180,
    n_rho0=None,
    delta_deg=None,
):
    """Coarse-to-fine Hough transform.

    Level 0:
        Build a coarse accumulator over the full space.

    Later levels:
        Refine only around the peaks detected at the
        previous level.
    """
    edges = np.asarray(
        edges,
        dtype=np.float64,
    )

    if edges.ndim != 2:
        raise ValueError(
            "edges must be a 2D array"
        )

    if n_levels <= 0:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    if refine_factor <= 1:
        raise ValueError(
            "refine_factor must be > 1"
        )

    # ------------------------------------------------------
    # Gradient direction
    # ------------------------------------------------------
    if grad_dir is not None:
        grad_dir_arr = np.asarray(
            grad_dir,
            dtype=np.float64,
        )

        if grad_dir_arr.shape != edges.shape:
            raise ValueError(
                "grad_dir must match edges shape"
            )
    else:
        grad_dir_arr = None

    # ------------------------------------------------------
    # Full rho range
    # ------------------------------------------------------
    H, W = edges.shape

    D = float(
        np.hypot(H, W)
    )

    if n_rho0 is None:
        n_rho0 = int(
            np.ceil(2 * D)
        ) + 1

    # ------------------------------------------------------
    # LEVEL 0: coarse Hough
    # ------------------------------------------------------
    acc, thetas, rhos = hough_accumulate(
        edges,
        theta_range=(0.0, 180.0),
        rho_range=(-D, D),
        n_theta=n_theta0,
        n_rho=n_rho0,
        grad_dir=grad_dir_arr,
        delta_deg=delta_deg,
    )

    peaks = find_peaks(
        acc,
        thetas=thetas,
        rhos=rhos,
        num_peaks=n_peaks,
        nms_radius=max(
            1,
            int(min(acc.shape) // 20),
        ),
        smooth_sigma=1.0,
    )

    if peaks.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    # ------------------------------------------------------
    # Initial resolution
    # ------------------------------------------------------
    if len(thetas) > 1:
        delta_theta = (
            np.pi / n_theta0
        )
    else:
        delta_theta = np.pi

    if len(rhos) > 1:
        delta_rho = (
            rhos[1] - rhos[0]
        )
    else:
        delta_rho = 1.0

    # ------------------------------------------------------
    # REFINEMENT LEVELS
    # ------------------------------------------------------
    for level in range(1, n_levels):

        refined_peaks = []

        # Each new level is refine_factor times finer.
        new_delta_theta = (
            delta_theta / refine_factor
        )

        new_delta_rho = (
            delta_rho / refine_factor
        )

        for rho_center, theta_center, _ in peaks:

            # --------------------------------------------------
            # Edge points that contributed to coarse peak
            # --------------------------------------------------
            y_idx, x_idx = np.nonzero(
                edges > 0
            )

            if y_idx.size == 0:
                continue

            coarse_rho = (
                x_idx * np.cos(theta_center)
                + y_idx * np.sin(theta_center)
            )

            candidate_mask = (
                np.abs(
                    coarse_rho - rho_center
                )
                <= delta_rho
            )

            selected_y = y_idx[
                candidate_mask
            ]

            selected_x = x_idx[
                candidate_mask
            ]

            if selected_y.size == 0:
                continue

            local_edges = np.zeros_like(
                edges,
                dtype=np.float64,
            )

            local_edges[
                selected_y,
                selected_x
            ] = 1.0

            # --------------------------------------------------
            # Handle theta wrap-around
            # --------------------------------------------------
            windows = _theta_windows(
                float(theta_center),
                float(delta_theta),
            )

            window_candidates = []

            for (
                theta_low,
                theta_high,
                rho_sign,
            ) in windows:

                theta_width = (
                    theta_high - theta_low
                )

                if theta_width <= 0:
                    continue

                n_theta_local = max(
                    3,
                    int(
                        np.ceil(
                            theta_width
                            / new_delta_theta
                        )
                    ) + 1,
                )

                # Convert to degrees for hough_accumulate
                low_deg = np.rad2deg(
                    theta_low
                )

                high_deg = np.rad2deg(
                    theta_high
                )

                rho_center_local = (
                    float(rho_center)
                    * rho_sign
                )

                rho_min_local = (
                    rho_center_local
                    - delta_rho
                )

                rho_max_local = (
                    rho_center_local
                    + delta_rho
                )

                n_rho_local = max(
                    3,
                    int(
                        np.ceil(
                            (rho_max_local
                             - rho_min_local)
                            / new_delta_rho
                        )
                    ) + 1,
                )

                grad_local = grad_dir_arr

                local_acc, local_thetas, local_rhos = (
                    hough_accumulate(
                        local_edges,
                        theta_range=(
                            low_deg,
                            high_deg,
                        ),
                        rho_range=(
                            rho_min_local,
                            rho_max_local,
                        ),
                        n_theta=n_theta_local,
                        n_rho=n_rho_local,
                        grad_dir=grad_local,
                        delta_deg=delta_deg,
                    )
                )

                local_peaks = find_peaks(
                    local_acc,
                    thetas=local_thetas,
                    rhos=local_rhos,
                    num_peaks=1,
                    nms_radius=1,
                    smooth_sigma=1.0,
                )

                if local_peaks.size:
                    rho_new = (
                        local_peaks[0, 0]
                        * rho_sign
                    )

                    theta_new = (
                        local_peaks[0, 1]
                    )

                    votes_new = (
                        local_peaks[0, 2]
                    )

                    window_candidates.append(
                        (
                            rho_new,
                            theta_new,
                            votes_new,
                        )
                    )

            if window_candidates:
                best = max(
                    window_candidates,
                    key=lambda p: p[2],
                )

                refined_peaks.append(
                    best
                )

        if not refined_peaks:
            break

        # --------------------------------------------------
        # Keep strongest peaks
        # --------------------------------------------------
        refined_peaks = sorted(
            refined_peaks,
            key=lambda p: p[2],
            reverse=True,
        )

        peaks = np.asarray(
            refined_peaks[:n_peaks],
            dtype=np.float64,
        )

        # Prepare finer resolution
        delta_theta = new_delta_theta
        delta_rho = new_delta_rho

    # ------------------------------------------------------
    # Return only (rho, theta)
    # ------------------------------------------------------
    if peaks.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    return np.asarray(
        peaks[:, :2],
        dtype=np.float64,
    )