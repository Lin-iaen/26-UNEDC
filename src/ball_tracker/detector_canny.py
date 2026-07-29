import cv2
import numpy as np


def detect(frame: np.ndarray,
           ball_radius_px_range: tuple[int, int] = (8, 40),
           pipe: dict | None = None,
           ) -> tuple[int | None, int | None,
                      int | None, np.ndarray | None]:
    """Canny + HoughCircles ball detector with multi-stage filtering.

    ``pipe`` is the dict returned by :func:`pipe_detector.detect`.

    Filter stages:
    1. HoughCircles finds candidate circles ordered by accumulator score
    2. Radius within expected range
    3. Center lies inside pipe ROI (if provided)
    4. Circle interior is significantly darker than exterior (colour check)
    5. Pick highest accumulator score among survivors
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

    edges = cv2.Canny(blurred, 30, 100)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=80, param2=25,
        minRadius=ball_radius_px_range[0],
        maxRadius=ball_radius_px_range[1]
    )

    if circles is None or len(circles[0]) == 0:
        return (None, None, None, edges)

    best_cx, best_cy, best_r = None, None, None
    best_contrast = 1e9

    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])

        # stage 2: radius sanity
        if r < ball_radius_px_range[0] or r > ball_radius_px_range[1]:
            continue

        # stage 3: pipe ROI gating
        if pipe is not None:
            mask = pipe["roi_mask"]
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                if mask[cy, cx] == 0:
                    continue
            else:
                continue

        # stage 4: colour contrast — interior must be darker than exterior
        h, w = frame.shape[:2]
        margin = r // 3
        iy0 = max(0, cy - r + margin)
        iy1 = min(h, cy + r - margin)
        ix0 = max(0, cx - r + margin)
        ix1 = min(w, cx + r - margin)
        interior = gray[iy0:iy1, ix0:ix1]
        if interior.size < 9:
            continue

        # exterior: ring just outside the circle
        ey0 = max(0, cy - r - 6)
        ey1 = min(h, cy + r + 6)
        ex0 = max(0, cx - r - 6)
        ex1 = min(w, cx + r + 6)
        outer_mask = np.zeros((ey1 - ey0, ex1 - ex0), dtype=np.uint8)
        cv2.circle(outer_mask,
                   (r + 6, r + 6) if cx - r - 6 >= 0 and cy - r - 6 >= 0
                   else (cx - ex0, cy - ey0),
                   r + 6, 255, -1)
        cv2.circle(outer_mask,
                   (cx - ex0, cy - ey0),
                   r, 0, -1)
        exterior = gray[ey0:ey1, ex0:ex1][outer_mask > 0]
        if exterior.size < 9:
            continue

        in_mean = float(np.mean(interior))
        out_mean = float(np.mean(exterior))
        contrast = in_mean - out_mean
        if contrast > -10:
            continue

        if contrast < best_contrast:
            best_contrast = contrast
            best_cx, best_cy, best_r = cx, cy, r

    return (best_cx, best_cy, best_r, edges)
