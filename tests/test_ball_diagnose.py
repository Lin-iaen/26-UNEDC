#!/usr/bin/env python3
"""Minimal diagnostic: test camera, MJPEG, and HDMI display independently.

Usage:
    source venv/bin/activate

    # Test MJPEG stream (view from browser at http://<pi-ip>:5000)
    python tests/test_ball_diagnose.py

    # Test cage display (wireless HDMI)
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER) \
        cage -- /home/lin/workspace/venv/bin/python tests/test_ball_diagnose.py --cage
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.drivers import Camera
from src.vision import MjpegStreamer

OUTPUT_W, OUTPUT_H = 640, 480
SENSOR_MODE = 1


def main():
    use_cage = "--cage" in sys.argv

    # ── camera ──────────────────────────────────────────────────────
    cam = Camera(vflip=True, hflip=True, output_size=(OUTPUT_W, OUTPUT_H))
    cam.start()

    # wait for first frame
    frame = None
    for _ in range(200):
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        print("ERROR: no camera frame")
        cam.release()
        sys.exit(1)
    print(f"Camera OK — {frame.shape[1]}x{frame.shape[0]}, frame_id={cam.frame_id}")

    # ── MJPEG stream (always starts, accessible from browser) ───────
    raw_provider_counter = {"n": 0, "last_id": -1}

    def raw_provider():
        fid = cam.frame_id
        if fid == raw_provider_counter["last_id"]:
            return None
        raw_provider_counter["last_id"] = fid
        return cam.read()

    streamer = MjpegStreamer(raw_provider, port=5000, max_fps=30)
    streamer.start()
    print(f"MJPEG: http://<pi-ip>:5000")

    # ── display ─────────────────────────────────────────────────────
    shared = {"overlay": None, "lock": threading.Lock()}

    def provider_annotated():
        with shared["lock"]:
            if shared["overlay"] is None:
                return cam.read()
            return shared["overlay"].copy()

    # Also serve annotated frames at port 5001 for comparison
    streamer2 = MjpegStreamer(
        provider_annotated, port=5001, max_fps=30,
        custom_template="""<!DOCTYPE html>
<html><body style="margin:0;background:#000">
<img src="/video_feed" style="width:100%%;object-fit:contain">
</body></html>""",
    )
    streamer2.start()
    print(f"MJPEG+overlay: http://<pi-ip>:5001")

    if use_cage:
        win_name = "Ball Diag"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    fps_c = 0
    fps_t = time.perf_counter()
    current_fps = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            out = frame.copy()
            cv2.putText(out, f"Diagnostic  {current_fps:.0f}fps  {w}x{h}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

            with shared["lock"]:
                shared["overlay"] = out

            if use_cage:
                cv2.imshow(win_name, out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            fps_c += 1
            elapsed = time.perf_counter() - fps_t
            if elapsed >= 1.0:
                current_fps = fps_c / elapsed
                fps_c = 0
                fps_t = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        print("Cleanup...")
        streamer.stop()
        streamer2.stop()
        if use_cage:
            cv2.destroyAllWindows()
        cam.release()


if __name__ == "__main__":
    main()
