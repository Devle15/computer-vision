import numpy as np

def clip_histogram(hist, clip_limit):
    """
    Clip histogram and redistribute excess counts.
    """
    hist = np.asarray(hist, dtype=np.int64)

    if hist.ndim != 1:
        raise ValueError("hist must be a 1D array")

    if hist.size == 0:
        return hist.copy()

    if clip_limit < 0:
        raise ValueError("clip_limit must be non-negative")

    # Histogram after clipping
    clipped = np.minimum(hist, clip_limit).astype(np.int64)

    # Total excess removed by clipping
    excess = hist - clipped
    total_excess = int(excess.sum())

    if total_excess == 0:
        return clipped

    L = clipped.size

    # Distribute evenly
    bonus = total_excess // L
    remainder = total_excess % L

    clipped += bonus

    # Distribute the remaining pixels one by one
    if remainder > 0:
        clipped[:remainder] += 1

    return clipped

def build_tile_luts(img, grid=(8, 8), clip_limit=2.0):
    raise NotImplementedError