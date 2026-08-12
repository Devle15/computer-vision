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


def hough_accumulate(edges, theta_range, rho_range,
                     n_theta=180, n_rho=None,
                     grad_dir=None, delta_deg=None):
    """Compute a line Hough accumulator for binary edge data."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 2:
        raise ValueError("edges must be a 2D array")
    if n_theta <= 0:
        raise ValueError("n_theta must be positive")

    theta_min, theta_max = _as_theta_range(theta_range)
    rho_min, rho_max = _as_rho_range(rho_range, edges)

    if n_rho is None:
        n_rho = max(1, int(np.ceil(np.hypot(edges.shape[0], edges.shape[1]))))
    n_rho = int(n_rho)
    if n_rho <= 0:
        raise ValueError("n_rho must be positive")

    thetas = np.deg2rad(np.linspace(theta_min, theta_max, n_theta, endpoint=False))
    thetas = _normalize_theta(thetas)
    thetas = np.sort(thetas)
    rhos = np.linspace(rho_min, rho_max, n_rho, endpoint=True)
    drho = (rho_max - rho_min) / n_rho if n_rho > 1 else 1.0
    acc = np.zeros((n_rho, n_theta), dtype=np.float64)

    grad_dir_arr = None
    if grad_dir is not None:
        grad_dir_arr = np.asarray(grad_dir, dtype=np.float64)
        if grad_dir_arr.shape != edges.shape:
            raise ValueError("grad_dir must have the same shape as edges")

    y_idx, x_idx = np.nonzero(edges > 0)
    if y_idx.size == 0:
        return acc, thetas, rhos

    for y, x in zip(y_idx, x_idx):
        if grad_dir_arr is not None:
            angle_deg = float(grad_dir_arr[y, x])
            orient = _normalize_theta(np.deg2rad(angle_deg))
            if delta_deg is not None:
                delta = np.deg2rad(float(delta_deg))
                valid = _angle_diff(thetas, orient) <= delta
                theta_candidates = thetas[valid]
                if theta_candidates.size == 0:
                    theta_candidates = thetas
            else:
                theta_candidates = thetas
        else:
            theta_candidates = thetas

        for theta in theta_candidates:
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_bin = int(np.clip(np.round((rho - rho_min) / drho), 0, n_rho - 1))
            theta_bin = int(np.clip(np.argmin(np.abs(_angle_diff(thetas, theta))), 0, n_theta - 1))
            acc[rho_bin, theta_bin] += 1.0

    return acc, thetas, rhos


def find_peaks(A, thetas=None, rhos=None, num_peaks=8, nms_radius=6, smooth_sigma=1.0):
    """Return the strongest peaks in a 2D accumulator as (rho, theta, votes)."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if thetas is None:
        thetas = np.arange(A.shape[1], dtype=np.float64)
    else:
        thetas = np.asarray(thetas, dtype=np.float64)
    if rhos is None:
        rhos = np.arange(A.shape[0], dtype=np.float64)
    else:
        rhos = np.asarray(rhos, dtype=np.float64)

    if num_peaks <= 0:
        return np.empty((0, 3), dtype=np.float64)

    filtered = _smooth_accumulator(A, float(smooth_sigma)) if smooth_sigma is not None else A.copy()
    radius = max(0, int(nms_radius))
    local_max = _local_maxima(filtered, radius)

    if not np.any(local_max):
        return np.empty((0, 3), dtype=np.float64)

    ys, xs = np.where(local_max)
    values = filtered[ys, xs]
    order = np.argsort(values)[::-1]

    selected = []
    for idx in order:
        y, x = ys[idx], xs[idx]
        if radius > 0 and any(abs(y - sy) <= radius and abs(x - sx) <= radius for sy, sx, _ in selected):
            continue
        selected.append((y, x, values[idx]))
        if len(selected) >= num_peaks:
            break

    peaks = []
    for y, x, value in selected:
        rho = rhos[int(y)]
        theta = thetas[int(x)]
        peaks.append((rho, theta, float(value)))
    return np.asarray(peaks, dtype=np.float64)


def hough_multiscale(edges, grad_dir=None,
                     n_peaks=8, n_levels=2,
                     refine_factor=8,
                     n_theta0=180, n_rho0=None,
                     delta_deg=None):
    """Run a coarse-to-fine Hough search and collect the strongest peaks."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 2:
        raise ValueError("edges must be a 2D array")
    if n_levels <= 0:
        return np.empty((0, 2), dtype=int)

    if n_rho0 is None:
        n_rho0 = max(1, int(np.ceil(np.hypot(edges.shape[0], edges.shape[1]))))

    all_peaks = []
    scores = []
    grad_level = None
    if grad_dir is not None:
        grad_level = np.asarray(grad_dir, dtype=np.float64)
        if grad_level.shape != edges.shape:
            raise ValueError("grad_dir must match the shape of edges")

    for level in range(n_levels):
        scale = 2 ** level
        if scale > 1:
            img = edges[::scale, ::scale]
            if grad_level is not None:
                grad_level_scaled = grad_level[::scale, ::scale]
            else:
                grad_level_scaled = None
        else:
            img = edges
            grad_level_scaled = grad_level

        theta_steps = max(10, int(n_theta0 // (refine_factor ** level)))
        rho_steps = max(1, int(n_rho0 // (refine_factor ** max(level, 1))))
        if rho_steps <= 0:
            rho_steps = n_rho0

        rho_min, rho_max = (-np.hypot(img.shape[0], img.shape[1]), np.hypot(img.shape[0], img.shape[1]))
        acc, thetas_acc, rhos_acc = hough_accumulate(
            img,
            theta_range=(-90.0, 90.0),
            rho_range=(rho_min, rho_max),
            n_theta=theta_steps,
            n_rho=rho_steps,
            grad_dir=grad_level_scaled,
            delta_deg=delta_deg
        )
        peaks = find_peaks(acc, thetas=thetas_acc, rhos=rhos_acc, num_peaks=max(1, int(n_peaks)), nms_radius=max(1, int(min(acc.shape) // 20)), smooth_sigma=1.0)
        if peaks.size == 0:
            continue

        for rho_peak, theta_peak, score in peaks:
            all_peaks.append((rho_peak, theta_peak))
            scores.append(score)

    if not all_peaks:
        return np.empty((0, 2), dtype=np.float64)

    order = np.argsort(scores)[::-1]
    best = np.asarray([all_peaks[i] for i in order[:max(1, int(n_peaks))]], dtype=np.float64)
    return best[:max(1, int(n_peaks))]