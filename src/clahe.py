import numpy as np


def clip_histogram(hist, clip_limit):
    # Tính phần dư
    excess_parts = np.maximum(hist - clip_limit, 0)
    total_excess = np.sum(excess_parts)
    
    # Cắt ngọn
    clipped_hist = np.minimum(hist, clip_limit)
    
    # Tính lượng chia đều và phần lẻ (remainder)
    L = len(hist) 
    bonus = total_excess // L
    remainder = total_excess % L
    
    # Phân phối phần chia đều
    final_hist = clipped_hist + bonus
    
    # Phân phối phần lẻ (tránh mất pixel)
    if remainder > 0:
        # Tính khoảng cách nhảy bước để rải đều phần dư
        step = L / remainder
        
        # Sinh ra các chỉ số cách đều nhau
        # Ví dụ L=4, remainder=2 => step=2.0 => indices=[0, 2]
        indices = (np.arange(remainder) * step).astype(int)
        
        # Cộng thêm 1 vào các vị trí đã chọn
        final_hist[indices] += 1
        
    return final_hist

if __name__ == "__main__":
    # TEST
    hist_test = np.array([10, 0, 0, 0])
    limit_test = 4

    print("Histogram gốc:", hist_test)
    print("Tổng ban đầu :", np.sum(hist_test))

    result = clip_histogram(hist_test, limit_test)

    print("Histogram sau:", result)
    print("Tổng lúc sau :", np.sum(result))

    # Kiểm tra tự động
    assert result.sum() == hist_test.sum(), f"Lỗi: Tổng sau xử lý ({result.sum()}) khác tổng ban đầu ({hist_test.sum()})"
    assert len(result) == len(hist_test), "Lỗi: Kích thước mảng bị thay đổi"
    print("=> TEST PASS! Tuyệt vời!")


def build_tile_luts(img, grid=(8, 8), clip_limit=2.0):
    raise NotImplementedError


def clahe(img, grid=(8, 8), clip_limit=2.0):
    raise NotImplementedError