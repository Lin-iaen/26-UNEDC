#!/usr/bin/env python3
"""IMX219 Camera Parameter Explorer (modular edition)
===================================================

Uses ``Camera`` and ``MjpegStreamer``.  All sliders auto-sync from the camera
state via the ``/stats`` endpoint — no stale defaults.

Usage:
    python tests/test_camera_params.py
    → open http://<pi-ip>:5000
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from flask import jsonify, request

from src.drivers import Camera
from src.vision import MjpegStreamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("camera_params")
logging.getLogger("picamera2").setLevel(logging.WARNING)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

vflip = True
hflip = False

_frame_count = 0
_fps = 0.0
_last_ts = time.perf_counter()

PRESETS = {
    "lowlight":     {"ExposureTime": 60000, "AnalogueGain": 8.0,  "Brightness": 0.3},
    "bright":       {"ExposureTime": 5000,  "AnalogueGain": 1.0,  "Brightness": -0.2},
    "default":      {"ExposureTime": 20000, "AnalogueGain": 1.0,  "Brightness": 0.0},
    "highcontrast": {"Contrast": 4.0, "Saturation": 2.0, "Sharpness": 4.0},
    "vivid":        {"Contrast": 2.0, "Saturation": 4.0, "Sharpness": 2.0},
}

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>IMX219 Explorer</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:360px;overflow-y:auto;padding:12px;background:#1a1a1a;border-right:1px solid #333}
img{max-width:100%;height:auto;display:block}
#main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:8px}
label{display:block;margin:8px 0 2px;font-size:12px;color:#aaa}
input[type=range]{width:100%}
.slider-row{display:flex;align-items:center;gap:8px}
.val{font-size:11px;color:#0f0;min-width:50px}
.btn{display:inline-block;margin:2px;padding:4px 10px;background:#333;color:#fff;border:1px solid #555;cursor:pointer;font-size:11px;text-decoration:none}
.btn:hover{background:#555}
h3{margin:12px 0 6px;font-size:13px;border-bottom:1px solid #333;padding-bottom:4px}
.stats{font-size:10px;color:#666;margin-top:4px}
</style></head><body>
<div id="panel">
<h3>IMX219 Controls</h3>

<label>ExposureTime <span class="val" id="v_ExposureTime">--</span></label>
<div class="slider-row">
<input type="range" id="sl_ExposureTime" min="39" max="66666" oninput="setVal('ExposureTime',this.value)">
<a class="btn" href="/reset/ae">AUTO</a>
</div>

<label>AnalogueGain <span class="val" id="v_AnalogueGain">--</span></label>
<div class="slider-row">
<input type="range" id="sl_AnalogueGain" min="1" max="16" step="0.1" oninput="setVal('AnalogueGain',this.value)">
</div>

<label>Brightness <span class="val" id="v_Brightness">--</span></label>
<div class="slider-row">
<input type="range" id="sl_Brightness" min="-1" max="1" step="0.05" oninput="setVal('Brightness',this.value)">
</div>

<label>Contrast <span class="val" id="v_Contrast">--</span></label>
<div class="slider-row">
<input type="range" id="sl_Contrast" min="0" max="32" step="0.1" oninput="setVal('Contrast',this.value)">
</div>

<label>Saturation <span class="val" id="v_Saturation">--</span></label>
<div class="slider-row">
<input type="range" id="sl_Saturation" min="0" max="32" step="0.1" oninput="setVal('Saturation',this.value)">
</div>

<label>Sharpness <span class="val" id="v_Sharpness">--</span></label>
<div class="slider-row">
<input type="range" id="sl_Sharpness" min="0" max="16" step="0.1" oninput="setVal('Sharpness',this.value)">
</div>

<label>ExposureValue <span class="val" id="v_ExposureValue">--</span></label>
<div class="slider-row">
<input type="range" id="sl_ExposureValue" min="-8" max="8" step="0.5" oninput="setVal('ExposureValue',this.value)">
</div>

<h3>Presets</h3>
<a class="btn" href="/set?preset=lowlight">Low Light</a>
<a class="btn" href="/set?preset=bright">Bright</a>
<a class="btn" href="/set?preset=default">Default</a>
<a class="btn" href="/set?preset=highcontrast">H Contrast</a>
<a class="btn" href="/set?preset=vivid">Vivid</a>

<h3>Sensor Modes</h3>
<div id="mode_buttons"></div>

<h3>Actions</h3>
<a class="btn" href="/capture">📷 Capture</a>
<a class="btn" href="/flip/v">↕ V-Flip</a>
<a class="btn" href="/flip/h">↔ H-Flip</a>

<div class="stats" style="margin-top:12px">
Frame: <span id="fps">--</span> FPS<br>
Mode: <span id="mode_label">--</span><br>
Timestamp: <span id="ts">--</span>
</div>
</div>
<div id="main">
<img src="/video_feed" id="stream">
</div>
<script>
const SLIDER_KEYS = ["ExposureTime","AnalogueGain","Brightness","Contrast","Saturation","Sharpness","ExposureValue"];
function setVal(name, v){
  document.getElementById('v_'+name).textContent=v;
  document.getElementById('sl_'+name).value=v;
  fetch('/set?'+name+'='+v);
}
function syncSliders(d){
  SLIDER_KEYS.forEach(function(k){
    var v = d[k.toLowerCase()] || d[k];
    if(v !== undefined && v !== null && v !== '--'){
      document.getElementById('v_'+k).textContent = v;
      document.getElementById('sl_'+k).value = v;
    }
  });
}
function poll(){
  fetch('/stats').then(r=>r.json()).then(function(d){
    syncSliders(d);
    document.getElementById('fps').textContent = d.fps;
    document.getElementById('ts').textContent = d.time;
    document.getElementById('mode_label').textContent = d.mode || '--';
  });
}
function initModeButtons(){
  fetch('/modes').then(r=>r.json()).then(function(modes){
    var html = '';
    modes.forEach(function(m,i){
      html += '<a class="btn" href="/mode/'+i+'">'+m.size[0]+'×'+m.size[1]+' @'+m.fps.toFixed(0)+'fps</a>';
    });
    document.getElementById('mode_buttons').innerHTML = html;
  });
}
initModeButtons();
poll();
setInterval(poll, 2000);
</script>
</body></html>"""


# ── Frame provider ───────────────────────────────────────────────────────────

def make_frame_provider(cam: Camera):
    latest_overlay: dict = {}

    def provider() -> np.ndarray | None:
        nonlocal latest_overlay
        global _frame_count, _fps, _last_ts

        frame = cam.read()
        if frame is None:
            return None

        if vflip and hflip:
            frame = cv2.flip(frame, -1)
        elif vflip:
            frame = cv2.flip(frame, 0)
        elif hflip:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"IMX219 {w}x{h}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        _frame_count += 1
        now = time.perf_counter()
        if now - _last_ts >= 1.0:
            _fps = _frame_count / (now - _last_ts)
            _frame_count = 0
            _last_ts = now

        return frame

    return provider, latest_overlay


# ── Custom route handlers ────────────────────────────────────────────────────

def make_route_set(cam: Camera, overlay: dict):
    def handler(**kwargs):
        preset = request.args.get("preset")
        if preset and preset in PRESETS:
            params = PRESETS[preset]
            logger.info("Preset: %s", preset)
        else:
            params = {}
            for key in request.args:
                if key == "preset":
                    continue
                try:
                    val = request.args[key]
                    params[key] = float(val) if "." in val else int(val)
                except ValueError:
                    pass
        cam.set_params(params)
        overlay.update(params)
        return jsonify({"ok": True, "values": params})
    return handler


def make_route_reset_ae():
    def handler(**kwargs):
        return jsonify({"ok": True})
    return handler


def make_route_capture(cam: Camera):
    def handler(**kwargs):
        frame = cam.read()
        if frame is None:
            return jsonify({"ok": False, "error": "no frame"}), 500
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SAMPLES_DIR / f"paramtest_{ts}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("Photo saved: %s", path)
        return jsonify({"ok": True, "path": str(path)})
    return handler


def make_route_flip():
    def handler(**kwargs):
        global vflip, hflip
        axis = kwargs.get("axis", "v")
        if axis == "v":
            vflip = not vflip
        elif axis == "h":
            hflip = not hflip
        return jsonify({"ok": True, "hflip": hflip, "vflip": vflip})
    return handler


def make_route_stats(overlay: dict, current_mode: list[int]):
    def handler(**kwargs):
        return jsonify({
            "fps": round(_fps, 1),
            "time": datetime.now().strftime("%H:%M:%S"),
            "mode": current_mode[0],
            "exposuretime": overlay.get("ExposureTime", ""),
            "analoguegain": overlay.get("AnalogueGain", ""),
            "brightness": overlay.get("Brightness", 0.0),
            "contrast": overlay.get("Contrast", 1.0),
            "saturation": overlay.get("Saturation", 1.0),
            "sharpness": overlay.get("Sharpness", 1.0),
            "exposurevalue": overlay.get("ExposureValue", 0.0),
        })
    return handler


def make_route_modes(cam: Camera):
    def handler(**kwargs):
        modes = []
        for m in cam.sensor_modes:
            modes.append({"size": list(m["size"]), "fps": m["fps"]})
        return jsonify(modes)
    return handler


def make_route_mode(cam: Camera, current_mode: list[int]):
    def handler(**kwargs):
        mode_id = int(kwargs.get("mode_id", 0))
        cam.switch_sensor_mode(mode_id)
        current_mode[0] = mode_id
        return jsonify({"ok": True, "mode": mode_id})
    return handler


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    cam = Camera(vflip=False, hflip=False)
    cam.start()

    provider, overlay = make_frame_provider(cam)
    current_mode = [0]  # mutable container so closures can update it

    # Seed overlay with initial defaults so sliders start at sane values
    overlay.update({
        "ExposureTime": 20000, "AnalogueGain": 1.0,
        "Brightness": 0.0, "Contrast": 1.0,
        "Saturation": 1.0, "Sharpness": 1.0, "ExposureValue": 0.0,
    })

    streamer = MjpegStreamer(
        frame_provider=provider,
        port=5000,
        custom_template=HTML_PAGE,
        custom_routes={
            "/set": make_route_set(cam, overlay),
            "/reset/ae": make_route_reset_ae(),
            "/capture": make_route_capture(cam),
            "/flip/<axis>": make_route_flip(),
            "/stats": make_route_stats(overlay, current_mode),
            "/modes": make_route_modes(cam),
            "/mode/<int:mode_id>": make_route_mode(cam, current_mode),
        },
    )
    streamer.start()
    logger.info("IMX219 Explorer ready at http://0.0.0.0:5000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down ...")

    streamer.stop()
    cam.release()
    logger.info("Done")


if __name__ == "__main__":
    main()
