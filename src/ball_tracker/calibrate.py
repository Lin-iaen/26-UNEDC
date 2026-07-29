import json
import math
import time
from pathlib import Path

CALIB_DIR = Path(__file__).resolve().parent.parent.parent / "calibration_data"
CALIB_FILE = CALIB_DIR / "ball_tracker.json"

DEFAULT_PIPE_LENGTH_MM = 250.0
LENS_FOV_DEG = 77.0


def _wait_for_ball(cam, detect_func, timeout: float = 15):
    """Read frames until detect_func locates the ball. Returns (x, y)."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        frame = cam.read()
        if frame is None:
            time.sleep(0.02)
            continue
        cx, cy, _, _ = detect_func(frame)
        if cx is not None:
            return (cx, cy)
    raise TimeoutError("Ball not detected — check lighting and camera focus")


# ── calibration methods ────────────────────────────────────────────────


def calibrate_by_ball_placement(cam, detect_func,
                                pipe_length_mm: float = DEFAULT_PIPE_LENGTH_MM) -> dict:
    print("\n=== Ball Placement Calibration ===")

    input("Step 1: Place ball at LEFT end of pipe, then press ENTER... ")
    print("  Detecting...")
    left = _wait_for_ball(cam, detect_func)
    print(f"  Left end: ({left[0]}, {left[1]})")

    input("Step 2: Place ball at RIGHT end of pipe, then press ENTER... ")
    print("  Detecting...")
    right = _wait_for_ball(cam, detect_func)
    print(f"  Right end: ({right[0]}, {right[1]})")

    pixel_dist = math.dist(left, right)
    scale = pipe_length_mm / pixel_dist
    print(f"\n  Distance: {pixel_dist:.0f} px  →  {scale:.4f} mm/px")

    angle_deg = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    return {
        "scale": scale,
        "unit": "mm/px",
        "method": "ball_placement",
        "pipe_length_mm": pipe_length_mm,
        "pipe_ends_px": [int(left[0]), int(left[1]),
                         int(right[0]), int(right[1])],
        "pipe_axis_deg": angle_deg,
    }


def calibrate_by_height_input(cam) -> dict:
    """Prompt for camera height via SSH, compute scale via FOV formula."""
    raw = input("Camera height above pipe (mm): ").strip()
    if not raw:
        raise RuntimeError("Height required")
    cam_height_mm = float(raw)

    for _ in range(50):
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        raise RuntimeError("No camera frame")
    img_w, img_h = frame.shape[1], frame.shape[0]

    diag_fov = math.radians(LENS_FOV_DEG)
    aspect = img_w / img_h
    diag_to_h = 1.0 / math.sqrt(1 + aspect * aspect)
    horiz_fov = 2 * math.atan(aspect * diag_to_h * math.tan(diag_fov / 2))
    horiz_width_mm = 2 * cam_height_mm * math.tan(horiz_fov / 2)
    scale = horiz_width_mm / img_w

    print(f"  Height={cam_height_mm:.0f}mm → FOV width={horiz_width_mm:.0f}mm"
          f" → {scale:.4f} mm/px")
    return {
        "scale": scale,
        "unit": "mm/px",
        "method": "fov_calc",
        "cam_height_mm": cam_height_mm,
        "pipe_length_mm": DEFAULT_PIPE_LENGTH_MM,
        "image_w": img_w,
        "image_h": img_h,
    }


# ── public API ─────────────────────────────────────────────────────────


def load() -> dict | None:
    if CALIB_FILE.exists():
        return json.loads(CALIB_FILE.read_text())
    return None


def save(calib: dict) -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps(calib, indent=2))
    print(f"Saved to {CALIB_FILE}")


def project_to_1d(x: int, y: int, calib: dict) -> float:
    pipe_ends = calib.get("pipe_ends_px")
    pipe_len_mm = calib.get("pipe_length_mm", 250.0)

    if pipe_ends and len(pipe_ends) == 4:
        p1x, p1y, p2x, p2y = pipe_ends
    else:
        h = calib.get("image_h", 480)
        w = calib.get("image_w", 640)
        p1x, p1y = 0, h // 2
        p2x, p2y = w, h // 2

    pipe_vx = p2x - p1x
    pipe_vy = p2y - p1y
    pipe_len_sq = pipe_vx * pipe_vx + pipe_vy * pipe_vy
    if pipe_len_sq < 1:
        return 0.0

    ball_vx = x - p1x
    ball_vy = y - p1y
    t = (ball_vx * pipe_vx + ball_vy * pipe_vy) / pipe_len_sq
    return max(0.0, min(1.0, t)) * pipe_len_mm


def run_interactive(cam, detect_func=None) -> dict:
    existing = load()
    if existing is not None:
        print(f"Calibration loaded from {CALIB_FILE}")
        return existing

    print("\nNo calibration found. Choose method:")
    print("  1 — Place ball at pipe ends (auto-detect, recommended)")
    print("  2 — Measure camera height (FOV calculation)")
    choice = input("Method [1]: ").strip() or "1"

    for _ in range(50):
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        raise RuntimeError("No camera frame")

    if choice == "1":
        if detect_func is None:
            print("No detector available — using height-based fallback.")
            return calibrate_by_height_input(cam)
        calib = calibrate_by_ball_placement(cam, detect_func, DEFAULT_PIPE_LENGTH_MM)
    elif choice == "2":
        calib = calibrate_by_height_input(cam)
    else:
        raise RuntimeError(f"Invalid choice: {choice}")

    calib.setdefault("image_w", frame.shape[1])
    calib.setdefault("image_h", frame.shape[0])
    save(calib)
    return calib
