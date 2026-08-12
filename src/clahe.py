import numpy as np


def clip_histogram(hist, clip_limit):
    # Tính phần dư
    excess_parts = np.maximum(hist - clip_limit, 0)
    
    # Tính tổng phần dư
    total_excess = np.sum(excess_parts)
    
    # Cắt ngọn
    clipped_hist = np.minimum(hist, clip_limit)
    
    # Tính lượng chia đều
    L = len(hist) 
    bonus = total_excess // L
    
    # Phân phối lại
    final_hist = clipped_hist + bonus
    
    return final_hist


def build_tile_luts(img, grid=(8, 8), clip_limit=2.0):
    raise NotImplementedError


def clahe(img, grid=(8, 8), clip_limit=2.0):
    raise NotImplementedError