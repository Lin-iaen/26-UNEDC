#!/usr/bin/env python3
"""CSI camera demo for Raspberry Pi (headless OpenCV compatible).

Usage:
    python src/camera_demo.py --capture           # Single photo
    python src/camera_demo.py --capture --vflip   #   with vertical flip
    python src/camera_demo.py --stream            # MJPEG HTTP stream at http://<ip>:5000
    python src/camera_demo.py --test 30           # Capture 30 frames, show FPS
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Run directly as `python src/camera_demo.py`: sys.path[0] is src/, not the
# project root, so `src.drivers` would not be importable without this.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drivers import Camera  # noqa: E402 — must follow the sys.path bootstrap

SAMPLES_DIR = PROJECT_ROOT / "samples"


def save_image(frame):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SAMPLES_DIR / f"capture_{ts}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {path}")


def capture_single(cam: Camera):
    frame = cam.read()
    if frame is not None:
        save_image(frame)
    else:
        print("No frame captured")


def run_test(cam: Camera, count: int, timeout: float = 5.0):
    """Measure real capture FPS.

    ``cam.read()`` hands back the cached frame, so looping on it alone measures
    memcpy speed (tens of thousands of "FPS"), not the sensor.  Wait for
    ``frame_id`` to advance so each counted frame is a genuinely new one.
    """
    print(f"Capturing {count} frames ...")
    captured = 0
    last_id = cam.frame_id
    last_new = time.perf_counter()
    start = time.perf_counter()

    while captured < count:
        current_id = cam.frame_id
        if current_id == last_id:
            if time.perf_counter() - last_new > timeout:
                print(f"  no new frame for {timeout:.0f}s — aborting")
                break
            time.sleep(0.001)
            continue

        last_id = current_id
        last_new = time.perf_counter()
        frame = cam.read()
        if frame is None:
            continue

        captured += 1
        if captured == count:
            save_image(frame)
        print(f"  {captured}/{count}  shape={frame.shape} dtype={frame.dtype}")

    elapsed = time.perf_counter() - start
    if captured == 0 or elapsed <= 0:
        print("\nResult: no frames captured")
        return
    print(f"\nResult: {captured} frames in {elapsed:.2f}s = {captured / elapsed:.1f} FPS")


def run_stream(cam: Camera, host: str = "0.0.0.0", port: int = 5000):
    from flask import Flask, Response

    app = Flask(__name__)

    def generate():
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(0.03)

    @app.route("/")
    def index():
        return '<img src="/video_feed" style="max-width:100%;max-height:100vh">'

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

    if not any([args.capture, args.stream, args.test is not None]):
        parser.print_help()
        return

    cam = Camera(vflip=args.vflip, hflip=args.hflip)
    try:
        cam.start()
        time.sleep(1.0)
        if args.capture:
            capture_single(cam)
        elif args.stream:
            run_stream(cam)
        elif args.test is not None:
            run_test(cam, args.test)
    finally:
        cam.release()


if __name__ == "__main__":
    main()
