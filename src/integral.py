import numpy as np


def integral_image(img):
    raise NotImplementedError


def box_sum(S, y1, x1, y2, x2):
    raise NotImplementedError


def local_mean_std(img, radius):
    raise NotImplementedError


def adaptive_threshold(img, radius=15, method="bradley", t=0.15, k=0.34):
    raise NotImplementedError