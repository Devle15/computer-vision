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
        return (-90.0, 90.0)
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
    drho = (rho_max - rho_min) / n_rho if n_rho > 1 else 1.0
    acc = np.zeros((n_rho, n_theta), dtype=np.float64)

    grad_dir_arr = None
    if grad_dir is not None:
        grad_dir_arr = np.asarray(grad_dir, dtype=np.float64)
        if grad_dir_arr.shape != edges.shape:
            raise ValueError("grad_dir must have the same shape as edges")

    y_idx, x_idx = np.nonzero(edges > 0)
    if y_idx.size == 0:
        return acc

    for y, x in zip(y_idx, x_idx):
        if grad_dir_arr is not None:
            angle_deg = float(grad_dir_arr[y, x])
            orient = np.deg2rad(angle_deg + 90.0)
            if delta_deg is not None:
                delta = np.deg2rad(float(delta_deg))
                valid = np.abs(np.angle(np.exp(1j * (thetas - orient)))) <= delta
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
            theta_bin = int(np.clip(np.argmin(np.abs(thetas - theta)), 0, n_theta - 1))
            acc[rho_bin, theta_bin] += 1.0

    return acc


def find_peaks(A, num_peaks=8, nms_radius=6, smooth_sigma=1.0):
    """Return the strongest local maxima in a 2D accumulator array."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if num_peaks <= 0:
        return np.empty((0, 2), dtype=int)

    if smooth_sigma is not None and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            filtered = gaussian_filter(A, sigma=float(smooth_sigma))
        except Exception:
            filtered = A.copy()
    else:
        filtered = A.copy()

    if nms_radius is None or nms_radius <= 0:
        local_max = filtered == filtered.max()
    else:
        try:
            from scipy.ndimage import maximum_filter
            radius = max(1, int(nms_radius))
            max_window = maximum_filter(filtered, size=2 * radius + 1, mode='constant')
            local_max = filtered == max_window
        except Exception:
            local_max = np.zeros_like(filtered, dtype=bool)
            radius = max(1, int(nms_radius))
            h, w = filtered.shape
            for i in range(h):
                i0 = max(0, i - radius)
                i1 = min(h, i + radius + 1)
                for j in range(w):
                    j0 = max(0, j - radius)
                    j1 = min(w, j + radius + 1)
                    patch = filtered[i0:i1, j0:j1]
                    local_max[i, j] = filtered[i, j] == patch.max()

    if not np.any(local_max):
        return np.empty((0, 2), dtype=int)

    ys, xs = np.where(local_max)
    values = filtered[ys, xs]
    order = np.argsort(values)[::-1]
    peak_idx = np.column_stack((ys[order], xs[order]))[:num_peaks]
    return peak_idx.astype(int)


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
        acc = hough_accumulate(
            img,
            theta_range=(-90.0, 90.0),
            rho_range=(rho_min, rho_max),
            n_theta=theta_steps,
            n_rho=rho_steps,
            grad_dir=grad_level_scaled,
            delta_deg=delta_deg
        )
        peaks = find_peaks(acc, num_peaks=max(1, int(n_peaks)), nms_radius=max(1, int(min(acc.shape) // 20)), smooth_sigma=1.0)
        if peaks.size == 0:
            continue

        for (rho_idx, theta_idx) in peaks:
            score = float(acc[rho_idx, theta_idx])
            all_peaks.append((rho_idx, theta_idx))
            scores.append(score)

    if not all_peaks:
        return np.empty((0, 2), dtype=int)

    order = np.argsort(scores)[::-1]
    best = np.asarray([all_peaks[i] for i in order[:max(1, int(n_peaks))]], dtype=int)
    return best[:max(1, int(n_peaks))]