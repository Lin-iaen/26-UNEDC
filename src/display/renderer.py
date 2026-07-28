"""OpenCV-based rendering for each UI state — (160, 128) portrait frames.

The display is physically rotated 90°, so portrait frames (160 rows ×
128 cols) drawn from "top to bottom" appear sideways on the landscape
panel.  Content is readable; only the layout direction is different.
"""

import cv2
import numpy as np

from .state_machine import MENU_ITEMS, HSV_PARAM_NAMES, UiState

W = 160
H = 128

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
CYAN = (230, 180, 0)
GRAY = (90, 90, 90)
DARK = (25, 25, 25)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 1
LINE_AA = cv2.LINE_AA

CAMERA_H = 128
INFO_H = H - CAMERA_H     # 32 px


# ── helpers ───────────────────────────────────────────────────────────

def _put(canvas, text, x, y, fg=WHITE, scale=None):
    s = scale or FONT_SCALE
    cv2.putText(canvas, text, (x, y), FONT, s, fg, FONT_THICK, LINE_AA)


def _put_bg(canvas, text, x, y, fg, bg, scale=None):
    sc = scale or FONT_SCALE
    sz, baseline = cv2.getTextSize(text, FONT, sc, FONT_THICK)
    tw, th = sz[0] + 1, sz[1] + baseline
    cv2.rectangle(canvas, (x - 2, y - th), (x + tw + 2, y + 2), bg, -1)
    cv2.putText(canvas, text, (x, y), FONT, sc, fg, FONT_THICK, LINE_AA)


# ── MENU renderer ─────────────────────────────────────────────────────

def render_menu(highlight_idx: int) -> np.ndarray:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    _put(canvas, "MENU", 40, 22, WHITE, 0.6)
    cv2.line(canvas, (10, 28), (118, 28), GRAY, 1)

    start_y = 48
    item_h = 26

    for i, label in enumerate(MENU_ITEMS):
        y = start_y + i * item_h
        text = f"{i + 1}.{label}"

        if i == highlight_idx:
            cv2.rectangle(canvas, (4, y - 10), (124, y + 10), CYAN, -1)
            _put(canvas, text, 10, y + 3, BLACK)
        else:
            _put(canvas, text, 10, y + 3, WHITE)

    _put(canvas, "A:Up B:Dn C:OK", 6, 155, GRAY, 0.38)
    return canvas


# ── DEBUG renderer ────────────────────────────────────────────────────

def _info_bar(canvas, values, cursor_idx, editing):
    """Draw 3-row × 2-column HSV table in bottom 32 px."""
    MIN_X = 14
    MAX_X = 72
    row_h = 9
    y0 = CAMERA_H + 8
    fs = 0.38

    for line in range(3):
        y = int(y0 + line * row_h)
        lo = HSV_PARAM_NAMES[line]
        hi = HSV_PARAM_NAMES[line + 3]
        lo_v = values.get(lo, 0)
        hi_v = values.get(hi, 0)

        lo_t = f"{lo[0].lower()}:{lo_v:3d}"
        hi_t = f"{hi[0].upper()}:{hi_v:3d}"

        if line == cursor_idx:
            fg_lo = fg_hi = CYAN
            bg = (20, 50, 80) if editing else None
        else:
            fg_lo = fg_hi = WHITE
            bg = None

        if bg:
            _put_bg(canvas, lo_t, MIN_X, y, fg_lo, bg, fs)
            _put_bg(canvas, hi_t, MAX_X, y, fg_hi, bg, fs)
        else:
            _put(canvas, lo_t, MIN_X, y, fg_lo, fs)
            _put(canvas, hi_t, MAX_X, y, fg_hi, fs)

    hint = "A- B+  C:OK  C-h:Save" if editing else "A:Bw B:Fd  C:Edit  C-h:Save"
    _put(canvas, hint, 4, H - 4, GRAY, 0.33)


def render_debug(camera_frame: np.ndarray | None,
                 param_values: dict[str, int],
                 param_index: int,
                 edit_mode: bool,
                 ) -> np.ndarray:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    if camera_frame is not None:
        f = camera_frame
        if f.shape[:2] != (CAMERA_H, W):
            f = cv2.resize(f, (W, CAMERA_H))
        canvas[:CAMERA_H, :] = f
    else:
        _put(canvas, "No Signal", 28, 56, GRAY, 0.55)

    canvas[CAMERA_H:, :] = DARK
    _info_bar(canvas, param_values, param_index, edit_mode)
    return canvas


# ── TASK renderer ─────────────────────────────────────────────────────

def render_task(task_name: str) -> np.ndarray:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    _put(canvas, task_name, 28, 50, GREEN, 0.6)
    _put(canvas, "Running...", 28, 74, WHITE, 0.45)
    _put(canvas, "C-h: Back", 6, 155, GRAY, 0.38)
    return canvas


# ── dispatch ──────────────────────────────────────────────────────────

def render_frame(
    state: UiState,
    menu_index: int = 0,
    camera_frame: np.ndarray | None = None,
    param_values: dict[str, int] | None = None,
    param_index: int = 0,
    edit_mode: bool = False,
) -> np.ndarray:
    if state == UiState.MENU:
        return render_menu(menu_index)
    elif state == UiState.DEBUG:
        return render_debug(camera_frame, param_values or {}, param_index, edit_mode)
    elif state in (UiState.TASK1, UiState.TASK2, UiState.TASK3):
        return render_task(state.name.replace("TASK", "Task "))
    return np.zeros((H, W, 3), dtype=np.uint8)
