import cv2
import numpy as np


def detect(frame: np.ndarray,
           pipe_roi: dict | None = None,
           ball_radius_px_range: tuple[int, int] = (8, 40),
           ) -> tuple[int | None, int | None,
                      int | None, np.ndarray | None]:
    """Canny + HoughCircles 检测，只接受落在管道ROI内的圆。

    过滤条件：
      1. 半径在 ball_radius_px_range 内
      2. 圆心在 pipe_roi 内（如果有）
      3. 圆内部比外部暗（颜色对比度）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if pipe_roi is not None and pipe_roi["y1"] < pipe_roi["y2"]:
        y1, y2 = pipe_roi["y1"], pipe_roi["y2"]
        roi_gray = gray[y1:y2, :]
        origin_y = y1
    else:
        roi_gray = gray
        y1 = 0
        origin_y = 0

    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 30, 100)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=80, param2=25,
        minRadius=ball_radius_px_range[0],
        maxRadius=ball_radius_px_range[1],
    )

    if circles is None or len(circles[0]) == 0:
        return (None, None, None, edges)

    best_cx, best_cy, best_r = None, None, None
    best_contrast = 1e9

    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])

        if r < ball_radius_px_range[0] or r > ball_radius_px_range[1]:
            continue

        # colour contrast: interior must be darker than exterior
        h_r, w_r = roi_gray.shape[:2]
        margin = r // 3
        iy0 = max(0, cy - r + margin)
        iy1 = min(h_r, cy + r - margin)
        ix0 = max(0, cx - r + margin)
        ix1 = min(w_r, cx + r - margin)
        interior = roi_gray[iy0:iy1, ix0:ix1]
        if interior.size < 9:
            continue

        ey0 = max(0, cy - r - 6)
        ey1 = min(h_r, cy + r + 6)
        ex0 = max(0, cx - r - 6)
        ex1 = min(w_r, cx + r + 6)
        outer_mask = np.zeros((ey1 - ey0, ex1 - ex0), dtype=np.uint8)
        cv2.circle(outer_mask, (cx - ex0, cy - ey0), r + 6, 255, -1)
        cv2.circle(outer_mask, (cx - ex0, cy - ey0), r, 0, -1)
        exterior = roi_gray[ey0:ey1, ex0:ex1][outer_mask > 0]
        if exterior.size < 9:
            continue

        in_mean = float(np.mean(interior))
        out_mean = float(np.mean(exterior))
        if in_mean - out_mean > -10:
            continue

        if (in_mean - out_mean) < best_contrast:
            best_contrast = in_mean - out_mean
            best_cx, best_cy, best_r = cx, cy + origin_y, r

    return (best_cx, best_cy, best_r, edges)
