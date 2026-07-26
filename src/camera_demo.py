#!/usr/bin/env python3
"""CSI camera demo for Raspberry Pi (headless OpenCV compatible).

Usage:
    python src/camera_demo.py --capture           # Single photo
    python src/camera_demo.py --capture --vflip   #   with vertical flip
    python src/camera_demo.py --stream            # MJPEG HTTP stream at http://<ip>:5000
    python src/camera_demo.py --test 30           # Capture 30 frames, show FPS
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def apply_flip(frame: np.ndarray, vflip: bool, hflip: bool) -> np.ndarray:
    if vflip and hflip:
        return cv2.flip(frame, -1)
    if vflip:
        return cv2.flip(frame, 0)
    if hflip:
        return cv2.flip(frame, 1)
    return frame


def capture_single(cam, vflip, hflip):
    cfg = cam.create_still_configuration()
    cam.configure(cfg)
    cam.start()
    frame = cam.capture_array()
    cam.stop()
    frame = apply_flip(frame, vflip, hflip)
    save_image(frame)


def save_image(frame):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SAMPLES_DIR / f"capture_{ts}.jpg"
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {path}")


def run_test(cam, count, vflip, hflip):
    cam.configure(cam.create_preview_configuration())
    cam.start()
    print(f"Capturing {count} frames ...")
    start = time.perf_counter()
    for i in range(count):
        frame = cam.capture_array()
        if i == count - 1:
            frame = apply_flip(frame, vflip, hflip)
            save_image(frame)
        print(f"  {i+1}/{count}  shape={frame.shape} dtype={frame.dtype}")
    elapsed = time.perf_counter() - start
    fps = count / elapsed
    print(f"\nResult: {count} frames in {elapsed:.2f}s = {fps:.1f} FPS")


def run_stream(cam, vflip, hflip, host="0.0.0.0", port=5000):
    from flask import Flask, Response

    cam.configure(cam.create_preview_configuration())
    cam.start()

    app = Flask(__name__)

    def generate():
        while True:
            frame = cam.capture_array()
            frame = apply_flip(frame, vflip, hflip)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(0.03)

    @app.route("/")
    def index():
        return '<img src="/video_feed" width="100%">'

    @app.route("/video_feed")
    def video_feed():
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"Stream ready at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


def main():
    parser = argparse.ArgumentParser(description="CSI Camera Demo")
    parser.add_argument("--capture", action="store_true", help="Capture single photo")
    parser.add_argument("--stream", action="store_true", help="Start MJPEG HTTP stream")
    parser.add_argument("--test", type=int, nargs="?", const=30, metavar="N",
                        help="Capture N frames and report FPS (default 30)")
    parser.add_argument("--vflip", action="store_true", help="Flip image vertically")
    parser.add_argument("--hflip", action="store_true", help="Flip image horizontally")
    args = parser.parse_args()

    if not any([args.capture, args.stream, args.test]):
        parser.print_help()
        return

    cam = Picamera2()
    try:
        if args.capture:
            capture_single(cam, args.vflip, args.hflip)
        elif args.stream:
            run_stream(cam, args.vflip, args.hflip)
        elif args.test:
            run_test(cam, args.test, args.vflip, args.hflip)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
