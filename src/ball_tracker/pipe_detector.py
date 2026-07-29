"""管道检测器 — Sobel 边缘投影法，不依赖颜色，抗白背景干扰。

原理：管壁的阴影边界产生强烈的垂直梯度（暗边→亮管）。
对每一行求和，梯度峰值对应管道上下边界。
"""

import cv2
import numpy as np


def detect(frame: np.ndarray) -> dict | None:
    """检测管道上下边界与轴线。

    管壁阴影在垂直方向产生强烈梯度，对水平方向求和的
    梯度轮廓取双峰 — 两峰即为管道上下边缘。
    返回管道方向包围盒或 None。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_sobel = np.abs(sobel_y)

    profile = np.sum(abs_sobel, axis=1).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (15, 1), 3).flatten()

    peaks = []
    for i in range(2, h - 2):
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        return None

    peaks_sorted = sorted(peaks, key=lambda i: profile[i], reverse=True)
    pipe_width_px = round(20.0 / 250.0 * w * 1.5)
    top_two = sorted(peaks_sorted[:4])

    best_pair = None
    best_dist = pipe_width_px * 2
    for i in range(len(top_two)):
        for j in range(i + 1, len(top_two)):
            dist = abs(top_two[j] - top_two[i])
            if abs(dist - pipe_width_px) < abs(best_dist - pipe_width_px):
                best_dist = dist
                best_pair = (top_two[i], top_two[j])

    if best_pair is None:
        return None

    y1, y2 = best_pair
    if y1 > y2:
        y1, y2 = y2, y1

    pipe_width = y2 - y1
    cy = (y1 + y2) // 2

    # Pipe axis: scan a few columns to estimate angle via fine-grained edge
    angles = []
    step = max(1, w // 16)
    for x in range(10, w - 10, step):
        col = gray[:, x].astype(np.float32)
        dy = np.gradient(col)
        dy_abs = np.abs(dy)
        if np.sum(dy_abs) < 1:
            continue
        local = peaks_sorted[:6]
        top_l = sorted(local)[:4]
        best_l = None
        best_d = pipe_width_px * 2
        for i in range(len(top_l)):
            for j in range(i + 1, len(top_l)):
                d = abs(top_l[j] - top_l[i])
                if abs(d - pipe_width_px) < abs(best_d - pipe_width_px):
                    best_d = d
                    best_l = (min(top_l[i], top_l[j]),
                              max(top_l[i], top_l[j]))
        if best_l is not None:
            angles.append(best_l[1])

    if len(angles) >= 4:
        pipe_width = round(np.mean(np.diff(np.sort(angles))))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (0, y1), (w, y2), 255, -1)

    # estimate axis angle from slope of best-fit line through midpoints
    midpoints = []
    step2 = max(1, w // 24)
    for x in range(0, w, step2):
        col = gray[:, x].astype(np.float32)
        grad = np.abs(np.gradient(col))
        max_row = y1 + np.argmax(grad[y1:y2 + 4]) if y1 < h else y1
        min_row = y1 + np.argmin(grad[y1:y2 + 4]) if y1 < h else y1
        midpoints.append((x, int((max_row + min_row) / 2)))

    if len(midpoints) >= 4:
        pts = np.array(midpoints, dtype=np.float32)
        vx, vy, cx_fit, cy_fit = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = float(np.degrees(np.arctan2(float(vy), float(vx))))
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        pipe_center = (round(float(cx_fit)), round(float(cy_fit)))
    else:
        angle = 0.0
        pipe_center = (w // 2, cy)

    return {
        "center_pt": (pipe_center[0], cy),
        "axis_angle": angle,
        "length_px": w,
        "width_px": pipe_width,
        "y_top": y1,
        "y_bottom": y2,
        "roi_mask": mask,
    }


def draw_overlay(frame: np.ndarray, pipe: dict) -> np.ndarray:
    out = frame.copy()
    if pipe is None:
        cv2.putText(out, "NO PIPE", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return out

    y1 = pipe["y_top"]
    y2 = pipe["y_bottom"]
    w = frame.shape[1]
    h = frame.shape[0]
    cy = pipe["center_pt"][1]

    cv2.rectangle(out, (0, y1), (w, y2), (255, 200, 0), 2)
    cv2.line(out, (0, cy), (w, cy), (0, 100, 255), 1)

    cv2.putText(out,
                f"pipe  W={pipe['width_px']}px  @{pipe['axis_angle']:.0f}deg",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return out
