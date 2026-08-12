import numpy as np


def convolve2d(img, kernel):
    raise NotImplementedError


def hough_accumulate(edges, theta_range, rho_range,
                     n_theta=180, n_rho=None,
                     grad_dir=None, delta_deg=None):
    """Accumulate votes for line candidates in Hough space.

    Parameters
    ----------
    edges : 2D array-like of bool/int
        Binary edge image.
    theta_range : (float, float)
        Range of line angles, in degrees or radians. Typical values are
        (0, 180) or (-90, 90) when supplied in degrees.
    rho_range : (float, float) or None
        Range of distances from the origin. If None, it is inferred from the
        image size.
    n_theta : int
        Number of bins along the theta axis.
    n_rho : int or None
        Number of bins along the rho axis. If None, infer from image diagonal.
    grad_dir : 2D array-like, optional
        Optional gradient direction map. If given, only cast votes whose
        orientation is consistent with the gradient direction within ``delta_deg``.
    delta_deg : float, optional
        Angular tolerance in degrees for the optional gradient filtering.

    Returns
    -------
    accumulator : 2D np.ndarray of shape (n_rho, n_theta)
        Vote counts in Hough parameter space.
    """
    edges = np.asarray(edges)
    if edges.ndim != 2:
        raise ValueError("edges must be a 2D array")

    if n_theta <= 0:
        raise ValueError("n_theta must be positive")
    if n_rho is None:
        n_rho = int(np.ceil(np.hypot(edges.shape[0], edges.shape[1]))) * 2 + 1
    if n_rho <= 0:
        raise ValueError("n_rho must be positive")

    if theta_range is None:
        theta_min, theta_max = 0.0, 180.0
    else:
        if not hasattr(theta_range, '__len__') or len(theta_range) != 2:
            raise ValueError("theta_range must be a length-2 sequence")
        theta_min, theta_max = float(theta_range[0]), float(theta_range[1])

    if rho_range is None:
        rho_min = -float(np.hypot(edges.shape[0], edges.shape[1]))
        rho_max = float(np.hypot(edges.shape[0], edges.shape[1]))
    else:
        if not hasattr(rho_range, '__len__') or len(rho_range) != 2:
            raise ValueError("rho_range must be a length-2 sequence")
        rho_min, rho_max = float(rho_range[0]), float(rho_range[1])

    if rho_min >= rho_max:
        raise ValueError("rho_range must satisfy rho_min < rho_max")

    theta_min_rad = theta_min
    theta_max_rad = theta_max
    if np.max(np.abs([theta_min_rad, theta_max_rad])) > 2.0 * np.pi + 1e-8:
        theta_min_rad = np.deg2rad(theta_min_rad)
        theta_max_rad = np.deg2rad(theta_max_rad)

    theta_vals = np.linspace(theta_min_rad, theta_max_rad, int(n_theta), endpoint=False)

    if grad_dir is not None:
        grad_dir = np.asarray(grad_dir, dtype=float)
        if grad_dir.shape != edges.shape:
            raise ValueError("grad_dir must have the same shape as edges")
        if np.max(np.abs(grad_dir)) > 2.0 * np.pi + 1e-8:
            grad_dir = np.deg2rad(grad_dir)

    tolerance = None if delta_deg is None else np.deg2rad(float(delta_deg))
    accumulator = np.zeros((int(n_rho), int(n_theta)), dtype=np.float64)

    ys, xs = np.nonzero(edges)
    if ys.size == 0:
        return accumulator

    for y, x in zip(ys, xs):
        if grad_dir is None:
            candidates = np.arange(theta_vals.size)
        else:
            direction = (grad_dir[y, x] + np.pi / 2.0) % np.pi
            if tolerance is None:
                candidates = np.arange(theta_vals.size)
            else:
                differences = np.abs((theta_vals - direction + np.pi) % (2.0 * np.pi) - np.pi)
                candidates = np.nonzero(differences <= tolerance)[0]

        for idx_theta in candidates:
            theta = theta_vals[idx_theta]
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_scale = rho_max - rho_min
            if rho_scale <= 0.0:
                continue
            rho_index = int(np.rint((rho - rho_min) / rho_scale * (n_rho - 1)))
            rho_index = max(0, min(n_rho - 1, rho_index))
            accumulator[rho_index, idx_theta] += 1.0

    return accumulator


def find_peaks(A, num_peaks=8, nms_radius=6, smooth_sigma=1.0):
    raise NotImplementedError


def hough_multiscale(edges, grad_dir=None,
                     n_peaks=8, n_levels=2,
                     refine_factor=8,
                     n_theta0=180, n_rho0=None,
                     delta_deg=None):
    raise NotImplementedError