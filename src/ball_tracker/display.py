import cv2
import numpy as np

DEBUG_RAW = 1
DEBUG_THRESH = 2
DEBUG_CANNY = 3
DEBUG_MASK = 4


def draw_overlay(frame: np.ndarray, x: int | None, y: int | None,
                 radius: int | None, pos_mm: float | None,
                 fps: float, debug_frame: np.ndarray | None,
                 debug_mode: int) -> np.ndarray:
    display = frame.copy()

    if debug_mode == DEBUG_RAW:
        pass
    elif debug_mode == DEBUG_THRESH:
        if debug_frame is not None:
            dbg = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR) if debug_frame.ndim == 2 else debug_frame
            display = cv2.resize(dbg, (frame.shape[1], frame.shape[0]))
    elif debug_mode == DEBUG_CANNY:
        if debug_frame is not None:
            dbg = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR) if debug_frame.ndim == 2 else debug_frame
            display = cv2.resize(dbg, (frame.shape[1], frame.shape[0]))
    elif debug_mode == DEBUG_MASK:
        if debug_frame is not None:
            dbg = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR) if debug_frame.ndim == 2 else debug_frame
            display = cv2.resize(dbg, (frame.shape[1], frame.shape[0]))

    if x is not None and y is not None and radius is not None:
        cv2.circle(display, (x, y), radius, (0, 255, 0), 2)
        cv2.line(display, (x - 8, y), (x + 8, y), (0, 255, 0), 1)
        cv2.line(display, (x, y - 8), (x, y + 8), (0, 255, 0), 1)
        cv2.circle(display, (x, y), 2, (0, 255, 0), -1)

    h, w = display.shape[:2]

    overlay_bar = display.copy()
    cv2.rectangle(overlay_bar, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bar, 0.5, display, 0.5, 0, display)

    lines = []
    mode_labels = {DEBUG_RAW: "RAW", DEBUG_THRESH: "THRESH",
                   DEBUG_CANNY: "CANNY", DEBUG_MASK: "MASK"}
    lines.append(f"[{mode_labels.get(debug_mode, '?')}]  {fps:.0f} fps")
    if pos_mm is not None:
        lines.append(f"pos: {pos_mm:.1f} mm")
    if x is not None:
        lines.append(f"center: ({x}, {y})  r={radius}px")

    for i, text in enumerate(lines):
        cv2.putText(display, text, (8, h - 32 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    return display


def mode_switch(key: int, current: int) -> int:
    if key == ord("1"):
        return DEBUG_RAW
    if key == ord("2"):
        return DEBUG_THRESH
    if key == ord("3"):
        return DEBUG_CANNY
    if key == ord("4"):
        return DEBUG_MASK
    return current
