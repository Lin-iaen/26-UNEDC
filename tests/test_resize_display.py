#!/usr/bin/env python3
"""Camera feed on HDMI with real‑time stretch/compress via web & keyboard.

运行方法:sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER)     cage -- /home/lin/workspace/venv/bin/python tests/test_resize_display.py

No physical keyboard needed — control width/height from your SSH terminal
or browser while the camera view fills the HDMI display.

Web control (open http://<pi-ip>:5001 in your laptop browser):
  ────────────────────────────────────────────
  ┃  Width  ┃████████████████████████████┃ 640
  ┃  Height ┃███████████████████████     ┃ 480
  ┃         [Reset] [Fill Display] [Quit]
  ────────────────────────────────────────────

Keyboard (if connected):
  a/d          width  −10 / +10          A/D       width  −100 / +100
  w/s          height +10 / −10          W/S       height +100 / −100
  r            reset                     f         fill display
  q / ESC      quit

Usage:
    # From SSH — web control works out of the box
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER) \
        cage -- /home/lin/workspace/venv/bin/python tests/test_resize_display.py

    # Start with custom initial size
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER) \
        cage -- /home/lin/workspace/venv/bin/python tests/test_resize_display.py \
            --init-width 320 --init-height 480

    # From your laptop, open:   http://<pi-ip>:5001
    # Or curl from SSH:         curl http://localhost:5001/set_size?w=800&h=200
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drivers import Camera

logging.getLogger("werkzeug").setLevel(logging.ERROR)

DEFAULT_DISPLAY_SIZE = (1024, 600)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Camera Resize Control</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
  h2 { margin-bottom: 8px; }
  .row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
  .row label { width: 60px; font-weight: bold; }
  input[type=range] { flex: 1; max-width: 500px; }
  .val { font-family: monospace; font-size: 1.2em; min-width: 50px; }
  .btn { padding: 8px 20px; border: none; border-radius: 4px; cursor: pointer;
         font-size: 1em; }
  .btn-primary { background: #0f3460; color: #eee; }
  .btn-danger  { background: #e94560; color: #fff; }
  .btn-success { background: #16c79a; color: #fff; }
  .info { margin-top: 16px; font-family: monospace; }
  #fps { color: #16c79a; }
</style>
</head>
<body>
<h2>Camera Resize</h2>
<div class="row">
  <label>Width</label>
  <input type="range" id="w" min="16" max="1920" value="640"
         oninput="setSize()">
  <span class="val" id="wv">640</span>
</div>
<div class="row">
  <label>Height</label>
  <input type="range" id="h" min="16" max="1080" value="480"
         oninput="setSize()">
  <span class="val" id="hv">480</span>
</div>
<div class="row">
  <button class="btn btn-primary" onclick="fetch('/reset')">Reset</button>
  <button class="btn btn-success" onclick="fetch('/full')">Fill Display</button>
  <button class="btn btn-danger"  onclick="fetch('/quit')">Quit</button>
</div>
<div class="info">
  FPS: <span id="fps">—</span> &nbsp;|&nbsp;
  Current: <span id="cur">—</span>
</div>
<script>
async function setSize() {
  const w = document.getElementById('w').value;
  const h = document.getElementById('h').value;
  document.getElementById('wv').textContent = w;
  document.getElementById('hv').textContent = h;
  const r = await fetch('/set_size?w='+w+'&h='+h).then(r => r.json());
  document.getElementById('cur').textContent = r.width + 'x' + r.height;
}
async function poll() {
  try {
    const r = await fetch('/status').then(r => r.json());
    document.getElementById('fps').textContent = r.fps.toFixed(1);
    document.getElementById('cur').textContent = r.width + 'x' + r.height;
    document.getElementById('w').value = r.width;
    document.getElementById('h').value = r.height;
    document.getElementById('wv').textContent = r.width;
    document.getElementById('hv').textContent = r.height;
  } catch(e) {}
  setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>"""


class SharedState:
    """Thread‑safe container for output dimensions and FPS."""

    def __init__(self, width: int, height: int):
        self._lock = threading.Lock()
        self.width = width
        self.height = height
        self.fps = 0.0

    def get(self) -> tuple[int, int]:
        with self._lock:
            return self.width, self.height

    def set(self, w: int, h: int):
        with self._lock:
            self.width = w
            self.height = h

    def adjust(self, dw: int, dh: int, min_val: int):
        with self._lock:
            self.width = max(self.width + dw, min_val)
            self.height = max(self.height + dh, min_val)

    def update_fps(self, fps: float):
        with self._lock:
            self.fps = fps

    def get_fps(self) -> float:
        with self._lock:
            return self.fps


def detect_display_size() -> tuple[int, int] | None:
    modes_glob = sorted(glob.glob("/sys/class/drm/*-HDMI-A-*/modes"))
    if not modes_glob:
        modes_glob = sorted(glob.glob("/sys/class/drm/*/modes"))
    for path in modes_glob:
        try:
            modes = Path(path).read_text().strip().split("\n")
            for mode in modes:
                mode = mode.strip()
                if not mode:
                    continue
                parts = mode.split("x")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
                    if w > 0 and h > 0:
                        return w, h
        except Exception:
            continue
    return None


def draw_overlay(frame: np.ndarray, width: int, height: int, fps: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bar_h = 50
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    text = f"{width} x {height}  |  FPS {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = h - bar_h // 2 + th // 2
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)


def parse_size(s: str):
    parts = s.split("x")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Camera feed on HDMI with real‑time resize (web + keyboard)"
    )
    parser.add_argument("--camera-size", default="640x480",
                        help="Camera output size (default: 640x480)")
    parser.add_argument("--init-width", type=int, default=None,
                        help="Initial display width (default: auto-detect)")
    parser.add_argument("--init-height", type=int, default=None,
                        help="Initial display height (default: auto-detect)")
    parser.add_argument("--min", type=int, default=16,
                        help="Minimum allowed width/height (default: 16)")
    parser.add_argument("--control-port", type=int, default=5001,
                        help="Web control panel port (default: 5001)")
    parser.add_argument("--no-vflip", action="store_true")
    parser.add_argument("--no-hflip", action="store_true")
    parser.add_argument("--exposure", type=int, default=30000)
    parser.add_argument("--gain", type=float, default=3.0)
    parser.add_argument("--fullscreen", action="store_true", default=True)
    parser.add_argument("--no-fullscreen", action="store_false", dest="fullscreen")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    if args.print_command:
        venv = Path(sys.executable).resolve()
        script = Path(__file__).resolve()
        print(
            f"sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER) "
            f"cage -- {venv} {script}"
        )
        return

    min_val = max(1, args.min)
    cam_w, cam_h = parse_size(args.camera_size)

    detected = detect_display_size()
    disp_w, disp_h = detected or DEFAULT_DISPLAY_SIZE

    out_w = args.init_width if args.init_width is not None else disp_w
    out_h = args.init_height if args.init_height is not None else disp_h
    out_w = max(out_w, min_val)
    out_h = max(out_h, min_val)

    state = SharedState(out_w, out_h)

    print(f"Camera: {cam_w}x{cam_h}")
    print(f"Display: {disp_w}x{disp_h}")
    print(f"Start size: {out_w}x{out_h}")
    print(f"Web control: http://<pi-ip>:{args.control_port}")

    # ── camera ──────────────────────────────────────────────────────
    cam = Camera(
        vflip=not args.no_vflip,
        hflip=not args.no_hflip,
        output_size=(cam_w, cam_h),
        exposure_time=args.exposure,
        analogue_gain=args.gain,
    )
    cam.start()

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

    # ── OpenCV window ───────────────────────────────────────────────
    win_name = "Camera HDMI"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win_name, out_w, out_h)

    # ── Flask control server ────────────────────────────────────────
    quit_flag = threading.Event()

    def start_web():
        from flask import Flask, jsonify, request

        app = Flask(__name__)

        @app.route("/")
        def index():
            return HTML_PAGE

        @app.route("/status")
        def status():
            w, h = state.get()
            return jsonify(width=w, height=h, fps=state.get_fps())

        @app.route("/set_size")
        def set_size():
            try:
                w = int(request.args.get("w", state.width))
                h = int(request.args.get("h", state.height))
                state.set(max(w, min_val), max(h, min_val))
            except Exception as exc:
                return jsonify(error=str(exc)), 400
            return jsonify(width=state.width, height=state.height)

        @app.route("/adjust")
        def adjust():
            dw = int(request.args.get("dw", 0))
            dh = int(request.args.get("dh", 0))
            state.adjust(dw, dh, min_val)
            return jsonify(width=state.width, height=state.height)

        @app.route("/reset")
        def reset():
            state.set(cam_w, cam_h)
            return jsonify(width=state.width, height=state.height)

        @app.route("/full")
        def full():
            state.set(disp_w, disp_h)
            return jsonify(width=state.width, height=state.height)

        @app.route("/quit")
        def quit_route():
            quit_flag.set()
            return "Quitting"

        from werkzeug.serving import make_server
        server = make_server("0.0.0.0", args.control_port, app, threaded=True)
        server.serve_forever()

    web_thread = threading.Thread(target=start_web, daemon=True)
    web_thread.start()

    # ── main display loop ──────────────────────────────────────────
    fps_counter = 0
    fps_time = time.perf_counter()

    try:
        while not quit_flag.is_set():
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            out_w, out_h = state.get()
            display = cv2.resize(frame, (out_w, out_h))
            draw_overlay(display, out_w, out_h, state.get_fps())
            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                state.adjust(-10, 0, min_val)
            elif key == ord("d"):
                state.adjust(10, 0, min_val)
            elif key == ord("w"):
                state.adjust(0, 10, min_val)
            elif key == ord("s"):
                state.adjust(0, -10, min_val)
            elif key == ord("A"):
                state.adjust(-100, 0, min_val)
            elif key == ord("D"):
                state.adjust(100, 0, min_val)
            elif key == ord("W"):
                state.adjust(0, 100, min_val)
            elif key == ord("S"):
                state.adjust(0, -100, min_val)
            elif key == ord("r"):
                state.set(cam_w, cam_h)
            elif key == ord("f"):
                state.set(disp_w, disp_h)

            fps_counter += 1
            elapsed = time.perf_counter() - fps_time
            if elapsed >= 1.0:
                state.update_fps(fps_counter / elapsed)
                fps_counter = 0
                fps_time = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        quit_flag.set()
        cv2.destroyAllWindows()
        cam.release()
        print("Done")


if __name__ == "__main__":
    main()
