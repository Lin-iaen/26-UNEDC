"""阈值检测器 — adaptiveThreshold + 形态学 + 面积筛选。

支持完整矩形 ROI [x1:x2, y1:y2]，检测结果限制在 ROI 内。
"""

import cv2
import numpy as np


def detect(frame: np.ndarray,
           pipe_roi: dict | None = None,
           block_size: int = 21,
           c_val: float = 4.0,
           morph_size: int = 5,
           min_area: float = 50.0,
           max_area: float = 800.0,
           ) -> tuple[int | None, int | None, int | None, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if pipe_roi is not None:
        x1 = max(0, pipe_roi.get("x1", 0))
        y1 = max(0, pipe_roi.get("y1", 0))
        x2 = min(frame.shape[1], pipe_roi.get("x2", frame.shape[1]))
        y2 = min(frame.shape[0], pipe_roi.get("y2", frame.shape[0]))
        if x1 < x2 and y1 < y2:
            roi_gray = gray[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            roi_gray = gray
            ox = oy = 0
    else:
        roi_gray = gray
        ox = oy = 0

    blurred = cv2.GaussianBlur(roi_gray, (7, 7), 1.5)

    blk = max(3, block_size) | 1
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blk, c_val,
    )

    ksize = max(1, morph_size) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = roi_gray.shape[:2]
    limit = max_area if max_area < h * w * 0.3 else h * w * 0.3
    best_cnt = None
    best_area = float("-inf")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > limit:
            continue
        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return (None, None, None, closed)

    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    cx, cy = int(cx) + ox, int(cy) + oy
    r = int(radius)

    # 边界校验：圆不能超出 ROI
    if pipe_roi is not None:
        if cx - r < x1 or cx + r > x2 or cy - r < y1 or cy + r > y2:
            return (None, None, None, closed)

    return (cx, cy, r, closed)
