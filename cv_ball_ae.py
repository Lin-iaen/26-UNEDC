#!/usr/bin/env python3
"""Ball-in-pipe tracker — auto exposure.

Usage:
    source venv/bin/activate

    # HDMI out (cage + wireless transmitter)
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER) \
        cage -- /path/to/venv/bin/python cv_ball_ae.py

    # Browser on http://<pi-ip>:5000 (no cage needed)
    python cv_ball_ae.py --web

Keyboard (HDMI mode):
    1/2/3/4   switch debug display mode
    t         toggle threshold / Canny+Hough detector
    q/ESC     quit
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from src.drivers import Camera
from src.vision import MjpegStreamer
from src.ball_tracker import calibrate, serial_out, display
from src.ball_tracker.calibrate import project_to_1d
from src.ball_tracker.detector_thresh import detect as detect_thresh
from src.ball_tracker.detector_canny import detect as detect_canny

SENSOR_MODE = 1
OUTPUT_W, OUTPUT_H = 640, 480
SERIAL_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
WEB_PORT = 5000

USE_THRESH = True

_latest_overlay = None
_overlay_lock = threading.Lock()


def _setup_cam() -> Camera:
    cam = Camera(
        vflip=True, hflip=True,
        output_size=(OUTPUT_W, OUTPUT_H),
    )
    cam.start()
    time.sleep(1.5)
    cam.switch_sensor_mode(SENSOR_MODE)
    cam.set_params({"AeEnable": True})
    return cam


def _wait_frame(cam: Camera, timeout: float = 8.0) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = cam.read()
        if frame is not None:
            return frame
        time.sleep(0.02)
    raise RuntimeError(f"No camera frame within {timeout}s")


def _make_provider(cam: Camera):
    counter = {"n": 0, "last_id": -1}

    def provider():
        with _overlay_lock:
            if _latest_overlay is not None:
                return _latest_overlay.copy()
        fid = cam.frame_id
        if fid == counter["last_id"]:
            return None
        counter["last_id"] = fid
        return cam.read()
    return provider


def _make_placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    return img


def main() -> None:
    use_web = "--web" in sys.argv

    cam = _setup_cam()
    frame0 = _wait_frame(cam)
    print(f"Camera: {frame0.shape[1]}x{frame0.shape[0]}  frame_id={cam.frame_id}")

    mjpeg = MjpegStreamer(
        frame_provider=_make_provider(cam),
        port=WEB_PORT, max_fps=30.0,
        custom_template="""<!DOCTYPE html>
<html><body style="margin:0;background:#000">
<img src="/video_feed" style="width:100%%;height:100%%;object-fit:contain">
</body></html>""",
    )
    mjpeg.start()
    print(f"MJPEG: http://<pi-ip>:{WEB_PORT}")

    if not use_web:
        win_name = "Ball Tracker [Auto Exposure]"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        placeholder = _make_placeholder(OUTPUT_W, OUTPUT_H,
                                        "Waiting for calibration")
        cv2.imshow(win_name, placeholder)
        cv2.waitKey(1)

    calib = calibrate.run_interactive(cam, detect_func=detect_thresh)

    ser = serial_out.SerialOut(SERIAL_PORT)
    print(f"Serial: {SERIAL_PORT} @ 115200")

    debug_mode = display.DEBUG_RAW
    use_thresh = USE_THRESH
    fps_c = 0
    fps_t = time.perf_counter()
    current_fps = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            cx, cy, radius, debug_img = (
                detect_thresh(frame) if use_thresh else detect_canny(frame)
            )

            pos_mm = None
            if cx is not None:
                pos_mm = project_to_1d(cx, cy, calib)
                ser.send(pos_mm)

            out = display.draw_overlay(frame, cx, cy, radius, pos_mm,
                                       current_fps, debug_img, debug_mode)

            with _overlay_lock:
                _latest_overlay = out

            if not use_web:
                cv2.imshow(win_name, out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("t"):
                    use_thresh = not use_thresh
                    print(f"Detector: {'threshold' if use_thresh else 'Canny+Hough'}")
                debug_mode = display.mode_switch(key, debug_mode)

            fps_c += 1
            elapsed = time.perf_counter() - fps_t
            if elapsed >= 1.0:
                current_fps = fps_c / elapsed
                fps_c = 0
                fps_t = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        mjpeg.stop()
        ser.close()
        cam.release()


if __name__ == "__main__":
    main()
