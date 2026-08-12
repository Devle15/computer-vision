import numpy as np


def convolve2d(img, kernel):
    raise NotImplementedError


def hough_accumulate(edges, theta_range, rho_range,
                     n_theta=180, n_rho=None,
                     grad_dir=None, delta_deg=None):
    raise NotImplementedError


def find_peaks(A, num_peaks=8, nms_radius=6, smooth_sigma=1.0):
    raise NotImplementedError


def hough_multiscale(edges, grad_dir=None,
                     n_peaks=8, n_levels=2,
                     refine_factor=8,
                     n_theta0=180, n_rho0=None,
                     delta_deg=None):
    raise NotImplementedError