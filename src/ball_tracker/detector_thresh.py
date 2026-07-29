import cv2
import numpy as np


def detect(frame: np.ndarray) -> tuple[int, int, int, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    img_area = h * w

    best = None
    best_circularity = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > img_area * 0.3:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.4:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0
        if fill_ratio < 0.3:
            continue

        if circularity > best_circularity:
            best_circularity = circularity
            best = (int(cx), int(cy), int(radius))

    if best is not None:
        return (*best, closed)
    return (None, None, None, closed)
