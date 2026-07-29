import cv2
import numpy as np


def detect(frame: np.ndarray,
           pipe_roi: dict | None = None,
           ) -> tuple[int | None, int | None,
                      int | None, np.ndarray | None]:
    """在管道ROI内找最暗连通块。

    丢掉 circularity / fill_ratio 检查，只按面积+亮度筛选。
    适用于运动模糊也稳定的策略 —— 管道内最暗的块只能是球。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if pipe_roi is not None and pipe_roi["y1"] < pipe_roi["y2"]:
        y1, y2 = pipe_roi["y1"], pipe_roi["y2"]
        roi_gray = gray[y1:y2, :]
    else:
        roi_gray = gray
        y1 = 0

    blurred = cv2.GaussianBlur(roi_gray, (7, 7), 1.5)

    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = roi_gray.shape[:2]
    max_area_limit = h * w * 0.3
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > max_area_limit:
            continue
        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return (None, None, None, closed)

    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    return (int(cx), int(cy) + y1, int(radius), closed)
