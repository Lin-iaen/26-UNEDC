import cv2
import numpy as np


def detect(frame: np.ndarray, ball_radius_px_range: tuple[int, int] = (8, 40)
            ) -> tuple[int, int, int, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

    edges = cv2.Canny(blurred, 30, 100)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=80, param2=25,
        minRadius=ball_radius_px_range[0],
        maxRadius=ball_radius_px_range[1]
    )

    if circles is not None and len(circles[0]) > 0:
        c = circles[0][0]
        return (int(c[0]), int(c[1]), int(c[2]), edges)
    return (None, None, None, edges)
