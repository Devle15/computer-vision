import numpy as np


def _compute_tile_edges(length, tiles):
    base = length // tiles
    remainder = length % tiles
    edges = [0]
    for index in range(tiles):
        edges.append(edges[-1] + base + (1 if index < remainder else 0))
    return np.array(edges, dtype=np.int32)


def clip_histogram(hist, clip_limit):
    hist = np.asarray(hist, dtype=np.int64).copy()
    if clip_limit is None or clip_limit <= 0:
        return hist

    clip_limit = int(np.floor(clip_limit))
    if clip_limit <= 0:
        return hist

    excess = np.maximum(hist - clip_limit, 0)
    total_excess = int(excess.sum())
    hist = np.minimum(hist, clip_limit)

    if total_excess == 0:
        return hist

    n_bins = hist.size
    while total_excess > 0:
        increment = total_excess // n_bins
        remainder = total_excess % n_bins
        if increment > 0:
            hist += increment
        if remainder > 0:
            hist[:remainder] += 1

        excess = np.maximum(hist - clip_limit, 0)
        total_excess = int(excess.sum())
        hist = np.minimum(hist, clip_limit)

    return hist


def build_tile_luts(img, grid=(8, 8), clip_limit=2.0):
    image = np.asarray(img)
    if image.ndim != 2:
        raise ValueError("CLAHE requires a 2D grayscale image")

    height, width = image.shape
    num_tiles_y, num_tiles_x = grid
    if num_tiles_y <= 0 or num_tiles_x <= 0:
        raise ValueError("Grid dimensions must be positive")

    row_edges = _compute_tile_edges(height, num_tiles_y)
    col_edges = _compute_tile_edges(width, num_tiles_x)
    n_bins = 256
    luts = np.zeros((num_tiles_y, num_tiles_x, n_bins), dtype=np.uint8)

    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            top, bottom = row_edges[tile_y], row_edges[tile_y + 1]
            left, right = col_edges[tile_x], col_edges[tile_x + 1]
            tile = image[top:bottom, left:right]
            if tile.size == 0:
                continue

            hist = np.bincount(tile.ravel(), minlength=n_bins).astype(np.int64)
            limit = int(np.floor(clip_limit * tile.size / n_bins)) if clip_limit > 0 else 0
            if limit > 0:
                hist = clip_histogram(hist, limit)

            cdf = hist.cumsum()
            nonzero = cdf > 0
            cdf_min = cdf[nonzero][0] if np.any(nonzero) else 0
            denom = tile.size - cdf_min
            if denom <= 0:
                lut = np.zeros(n_bins, dtype=np.uint8)
            else:
                lut = np.floor((cdf - cdf_min) / denom * (n_bins - 1)).astype(np.uint8)

            luts[tile_y, tile_x] = lut

    return luts


def clahe(img, grid=(8, 8), clip_limit=2.0):
    image = np.asarray(img)
    if image.ndim != 2:
        raise ValueError("CLAHE requires a 2D grayscale image")

    height, width = image.shape
    num_tiles_y, num_tiles_x = grid
    if num_tiles_y <= 0 or num_tiles_x <= 0:
        raise ValueError("Grid dimensions must be positive")

    row_edges = _compute_tile_edges(height, num_tiles_y)
    col_edges = _compute_tile_edges(width, num_tiles_x)
    luts = build_tile_luts(image, grid=grid, clip_limit=clip_limit)
    output = np.zeros_like(image, dtype=np.uint8)

    row_indices = np.arange(height)
    col_indices = np.arange(width)
    tile_rows = np.searchsorted(row_edges, row_indices, side="right") - 1
    tile_cols = np.searchsorted(col_edges, col_indices, side="right") - 1
    tile_rows = np.clip(tile_rows, 0, num_tiles_y - 1)
    tile_cols = np.clip(tile_cols, 0, num_tiles_x - 1)

    row_low = row_edges[tile_rows]
    row_high = row_edges[tile_rows + 1]
    row_span = np.maximum(row_high - row_low, 1)
    row_weight = (row_indices - row_low) / row_span

    col_low = col_edges[tile_cols]
    col_high = col_edges[tile_cols + 1]
    col_span = np.maximum(col_high - col_low, 1)
    col_weight = (col_indices - col_low) / col_span

    for r in range(height):
        i = tile_rows[r]
        i1 = min(i + 1, num_tiles_y - 1)
        t = float(row_weight[r])
        inv_t = 1.0 - t

        for c in range(width):
            j = tile_cols[c]
            j1 = min(j + 1, num_tiles_x - 1)
            s = float(col_weight[c])
            inv_s = 1.0 - s

            intensity = int(image[r, c])
            v00 = int(luts[i, j, intensity])
            v01 = int(luts[i, j1, intensity])
            v10 = int(luts[i1, j, intensity])
            v11 = int(luts[i1, j1, intensity])

            if i == i1 and j == j1:
                output[r, c] = v00
            elif i == i1:
                value = inv_s * v00 + s * v01
                output[r, c] = np.uint8(np.round(value))
            elif j == j1:
                value = inv_t * v00 + t * v10
                output[r, c] = np.uint8(np.round(value))
            else:
                value = (
                    inv_t * inv_s * v00
                    + inv_t * s * v01
                    + t * inv_s * v10
                    + t * s * v11
                )
                output[r, c] = np.uint8(np.round(value))

    return output
