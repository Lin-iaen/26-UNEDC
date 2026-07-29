#!/usr/bin/env python3
"""Ball-in-pipe tracker — fixed exposure + gain.

Usage:
    source venv/bin/activate
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER) \
        cage -- ./venv/bin/python cv_ball_fixed.py
    python cv_ball_fixed.py --web

Keyboard: 1/2/3/4 debug mode, t toggle detector, q quit
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
EXPOSURE_US = 15000
GAIN = 3.0
SERIAL_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
WEB_PORT = 5000
HOLD_FRAMES = 5  # number of frames to keep last valid position on miss

USE_THRESH = True

_latest_overlay = None
_overlay_lock = threading.Lock()


def _setup_cam():
    cam = Camera(vflip=True, hflip=True,
                 exposure_time=EXPOSURE_US, analogue_gain=GAIN,
                 output_size=(OUTPUT_W, OUTPUT_H))
    cam.start()
    time.sleep(1.5)
    cam.switch_sensor_mode(SENSOR_MODE)
    cam.set_params({"AeEnable": False})
    return cam


def _wait_frame(cam, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = cam.read()
        if frame is not None:
            return frame
        time.sleep(0.02)
    raise RuntimeError(f"No frame within {timeout}s")


def _make_provider(cam):
    counter = {"n": 0, "last_id": -1}
    def provider():
        fid = cam.frame_id
        if fid == counter["last_id"]:
            return None
        counter["last_id"] = fid
        with _overlay_lock:
            if _latest_overlay is not None:
                return _latest_overlay.copy()
        return cam.read()
    return provider


def main():
    use_web = "--web" in sys.argv

    cam = _setup_cam()
    frame0 = _wait_frame(cam)
    print(f"Camera: {frame0.shape[1]}x{frame0.shape[0]}  frame_id={cam.frame_id}")

    calib = calibrate.run_interactive(cam, detect_func=detect_thresh)

    # read pipe ROI from calibration
    roi = None
    if calib and "pipe_roi_y1" in calib and "pipe_roi_y2" in calib:
        y1 = int(calib["pipe_roi_y1"])
        y2 = int(calib["pipe_roi_y2"])
        x1 = int(calib.get("pipe_roi_x1", 0))
        x2 = int(calib.get("pipe_roi_x2", frame0.shape[1] - 1))
        if 0 <= y1 < y2 <= frame0.shape[0] and 0 <= x1 < x2:
            roi = {"y1": y1, "y2": y2, "x1": x1, "x2": x2}
            print(f"Pipe ROI: x{x1}-x{x2} y{y1}-y{y2}")

    ser = serial_out.SerialOut(SERIAL_PORT)
    print(f"Serial: {SERIAL_PORT} @ 115200")

    mjpeg = MjpegStreamer(
        frame_provider=_make_provider(cam),
        port=WEB_PORT, max_fps=30.0,
    )
    mjpeg.start()
    print(f"MJPEG: http://<pi-ip>:{WEB_PORT}")

    if not use_web:
        win = "Ball Tracker [Fixed]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    debug_mode = display.DEBUG_RAW
    use_thresh = USE_THRESH
    fps_c = 0
    fps_t = time.perf_counter()
    current_fps = 0.0

    # frame persistence
    hold = {"x": None, "y": None, "age": 99}

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            cx, cy, radius, debug_img = (
                detect_thresh(frame, pipe_roi=roi)
                if use_thresh else detect_canny(frame, pipe_roi=roi)
            )

            # frame persistence: keep last position on brief misses
            if cx is not None:
                hold["x"], hold["y"] = cx, cy
                hold["age"] = 0
            elif hold["age"] < HOLD_FRAMES:
                cx, cy = hold["x"], hold["y"]
                hold["age"] += 1
            else:
                hold["x"] = hold["y"] = None

            pos_mm = None
            if cx is not None:
                pos_mm = project_to_1d(cx, cy, calib)
                ser.send(pos_mm)

            # draw ROI rectangle
            if roi:
                x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            out = display.draw_overlay(frame, cx, cy, radius, pos_mm,
                                       current_fps, debug_img, debug_mode)

            with _overlay_lock:
                _latest_overlay = out

            if not use_web:
                cv2.imshow(win, out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("t"):
                    use_thresh = not use_thresh
                    print(f"Detector: {'thresh' if use_thresh else 'canny'}")
                debug_mode = display.mode_switch(key, debug_mode)

            fps_c += 1
            if time.perf_counter() - fps_t >= 1.0:
                current_fps = fps_c / (time.perf_counter() - fps_t)
                fps_c, fps_t = 0, time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        mjpeg.stop()
        ser.close()
        cam.release()


if __name__ == "__main__":
    main()
