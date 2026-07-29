#!/usr/bin/env python3
"""阈值调参工具 —— 实时调节检测参数，边调边看效果。

用法：
    source venv/bin/activate
    python tests/test_threshold_tuner.py
    → 浏览器打开 http://<pi-ip>:5002

也可以在 cage 下 HDMI 输出：
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u $USER) \
        cage -- /home/lin/workspace/venv/bin/python tests/test_threshold_tuner.py --cage

支持两种检测器：
  自适应阈值 (thresh) — blockSize, C, morph_size, min_area, circularity
  Canny+Hough (canny) — low, high, minR, maxR
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from flask import jsonify, request

from src.drivers import Camera
from src.vision import MjpegStreamer

SENSOR_MODE = 1
OUTPUT_W, OUTPUT_H = 640, 480
TUNER_PORT = 5002

# ── 可调参数（min, max, step, 默认值）────────────────────────────

THRESH_PARAMS = {
    "blockSize": (11, 51, 2, 21),
    "C": (2, 20, 1, 4),
    "morph_size": (1, 15, 2, 5),
    "min_area": (10, 500, 10, 50),
    "circularity": (0.1, 1.0, 0.05, 0.4),
}

CANNY_PARAMS = {
    "low": (10, 200, 5, 30),
    "high": (30, 400, 10, 100),
    "minR": (5, 50, 1, 8),
    "maxR": (20, 100, 1, 40),
}

STATE = {
    "detector": "thresh",
    "params": {k: v[3] for k, v in THRESH_PARAMS.items()},
    "detected": "-",
    "fps": 0.0,
}


def apply_thresh(frame: np.ndarray, p: dict) -> tuple[np.ndarray | None,
                                                       int | None, int | None, int | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blk = int(p["blockSize"]) | 1
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blk, float(p["C"])
    )

    ksize = int(p["morph_size"]) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    img_area = h * w
    min_a = float(p["min_area"])
    circ = float(p["circularity"])
    best, best_circ = None, 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_a or area > img_area * 0.3:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < circ:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        ca = np.pi * radius * radius
        if ca > 0 and area / ca < 0.3:
            continue
        if circularity > best_circ:
            best_circ = circularity
            best = (int(cx), int(cy), int(radius))

    return closed, *(best or (None, None, None))


def apply_canny(frame: np.ndarray, p: dict) -> tuple[np.ndarray | None,
                                                      int | None, int | None, int | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

    edges = cv2.Canny(blurred, int(p["low"]), int(p["high"]))

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=int(p["high"]), param2=25,
        minRadius=int(p["minR"]), maxRadius=int(p["maxR"]),
    )

    result = edges
    if circles is not None and len(circles[0]) > 0:
        c = circles[0][0]
        return result, int(c[0]), int(c[1]), int(c[2])
    return result, None, None, None


DETECTORS = {"thresh": apply_thresh, "canny": apply_canny}
PARAM_SETS = {"thresh": THRESH_PARAMS, "canny": CANNY_PARAMS}


def make_frame_provider(cam: Camera):
    counter = {"n": 0, "last_id": -1}

    def provider():
        nonlocal counter
        fid = cam.frame_id
        if fid == counter["last_id"]:
            return None
        counter["last_id"] = fid

        frame = cam.read()
        if frame is None:
            return None

        det = STATE["detector"]
        p = STATE["params"]
        proc, cx, cy, r = DETECTORS[det](frame, p)

        h, w = frame.shape[:2]

        # side-by-side: left = original + overlay, right = processed
        if proc is not None:
            if proc.ndim == 2:
                proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            else:
                proc_bgr = proc
            proc_bgr = cv2.resize(proc_bgr, (w, h))
        else:
            proc_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        out = np.hstack((frame, proc_bgr))

        if cx is not None:
            for arr, ox in [(frame, 0), (proc_bgr, w)]:
                cv2.circle(arr, (cx, cy), r, (0, 255, 0), 2)
                cv2.line(arr, (cx - 6, cy), (cx + 6, cy), (0, 255, 0), 1)
                cv2.line(arr, (cx, cy - 6), (cx, cy + 6), (0, 255, 0), 1)

        label = f"[{det}]  ({cx},{cy}) r={r}" if cx else f"[{det}]  no det"
        STATE["detected"] = label

        cv2.putText(out, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(out, f"{w}x{h}", (out.shape[1] - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return out

    return provider


# ── HTML + JS ────────────────────────────────────────────────────────

def build_page() -> str:
    def _slider_html(items, det):
        rows = []
        for k, v in items:
            rows.append(
                f'<div class="slider-row">'
                f'<label>{k} <span id="{det[0]}v_{k}" class="val"></span></label>'
                f'<input type="range" id="{det[0]}s_{k}" min="{v[0]}" max="{v[1]}" '
                f'step="{v[2]}" value="{v[3]}" '
                f'oninput="setP(\'{det}\',\'{k}\',this.value)">'
                f'</div>'
            )
        return "".join(rows)

    thresh_sliders = _slider_html(THRESH_PARAMS.items(), "thresh")
    canny_sliders = _slider_html(CANNY_PARAMS.items(), "canny")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>阈值调参工具</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}}
#panel{{width:380px;overflow-y:auto;padding:12px;background:#1a1a1a;border-right:1px solid #333}}
#main{{flex:1;display:flex;align-items:center;justify-content:center;background:#000}}
img{{max-width:100%;max-height:100vh;object-fit:contain}}
h3{{margin:12px 0 6px;font-size:12px;color:#6cf;border-bottom:1px solid #333;padding-bottom:3px}}
.slider-row{{margin:6px 0}}
.slider-row label{{display:block;font-size:11px;color:#aaa;margin-bottom:2px}}
.slider-row input[type=range]{{width:100%}}
.val{{float:right;color:#0f0;font-size:11px}}
.btn{{display:inline-block;margin:3px 2px;padding:5px 9px;background:#2a2a2a;color:#ddd;
      border:1px solid #444;cursor:pointer;font-size:11px;font-family:monospace}}
.btn:hover{{background:#3a3a3a}}
.btn.on{{background:#1a3a1a;border-color:#2a6a2a;color:#8f8}}
#det_info{{font-size:11px;color:#fa0;margin-top:8px;min-height:14px}}
.hint{{font-size:10px;color:#777;margin:4px 0}}
</style></head><body>
<div id="panel">
<h3>检测器选择</h3>
<span class="btn on" id="b_thresh" onclick="sw('thresh')">自适应阈值</span>
<span class="btn" id="b_canny" onclick="sw('canny')">Canny+Hough</span>

<h3>自适应阈值参数</h3>
<div id="thresh_panel">{thresh_sliders}</div>

<h3>Canny+Hough 参数</h3>
<div id="canny_panel">{canny_sliders}</div>

<div id="det_info">--</div>
</div>
<div id="main"><img src="/video_feed"></div>

<script>
var active = 'thresh';

function showPanel(det){{
  document.getElementById('thresh_panel').style.display = det==='thresh' ? 'block' : 'none';
  document.getElementById('canny_panel').style.display = det==='canny' ? 'block' : 'none';
  document.getElementById('b_thresh').className = 'btn' + (det==='thresh'?' on':'');
  document.getElementById('b_canny').className = 'btn' + (det==='canny'?' on':'');
}}

var pending = {{}}, timers = {{}};
function setP(det, key, val){{
  document.getElementById((det==='thresh'?'tv_':'cv_')+key).textContent = val;
  pending[key] = val;
  if(timers[key]) return;
  timers[key] = setTimeout(function(){{
    timers[key] = null;
    var q = det+'?'+key+'='+pending[key];
    fetch('/set?'+q).catch(e=>{{}});
  }}, 60);
}}

function sw(det){{
  active = det;
  showPanel(det);
  fetch('/set?detector='+det).catch(e=>{{}});
}}

function poll(){{
  fetch('/stats').then(r=>r.json()).then(function(d){{
    document.getElementById('det_info').textContent = d.detected + '  ' + d.fps + ' fps';
    for(var k in d.params){{
      var el = document.getElementById((active==='thresh'?'tv_':'cv_')+k);
      if(el) el.textContent = d.params[k];
    }}
  }}).catch(e=>{{}});
}}

showPanel('thresh');
poll();
setInterval(poll, 1000);
</script>
</body></html>"""


# ── 路由 ──────────────────────────────────────────────────────────────

def make_routes():
    def route_stats(**kw):
        return jsonify({
            "detected": STATE["detected"],
            "fps": STATE["fps"],
            "params": STATE["params"],
        })

    def route_set(**kw):
        for key in request.args:
            if key == "detector":
                det = request.args[key]
                if det in DETECTORS:
                    STATE["detector"] = det
                    STATE["params"] = {k: v[3] for k, v in PARAM_SETS[det].items()}
                continue
            if key in STATE["params"]:
                try:
                    STATE["params"][key] = float(request.args[key])
                except ValueError:
                    pass
        return jsonify({"ok": True})

    return {"/stats": route_stats, "/set": route_set}


# ── 入口 ──────────────────────────────────────────────────────────────

def main():
    use_cage = "--cage" in sys.argv

    cam = Camera(vflip=True, hflip=True, output_size=(OUTPUT_W, OUTPUT_H))
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

    print(f"Camera: {frame.shape[1]}x{frame.shape[0]}")
    cam.switch_sensor_mode(SENSOR_MODE)
    cam.set_params({"AeEnable": True})

    streamer = MjpegStreamer(
        frame_provider=make_frame_provider(cam),
        port=TUNER_PORT, max_fps=30.0,
        custom_template=build_page(),
        custom_routes=make_routes(),
    )
    streamer.start()
    print(f"阈值调参工具 → http://<pi-ip>:{TUNER_PORT}")

    if use_cage:
        win_name = "Threshold Tuner"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        # show placeholder
        placeholder = np.zeros((OUTPUT_H * 2, OUTPUT_W * 2, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Threshold Tuner — use browser at :5002",
                    (10, OUTPUT_H), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (100, 100, 100), 1)
        cv2.imshow(win_name, placeholder)
        cv2.waitKey(1)
    else:
        win_name = None

    fps_c, fps_t = 0, time.perf_counter()

    try:
        while True:
            if use_cage:
                cv2.waitKey(50)
            else:
                time.sleep(0.05)

            fps_c += 1
            elapsed = time.perf_counter() - fps_t
            if elapsed >= 1.0:
                STATE["fps"] = round(fps_c / elapsed, 1)
                fps_c, fps_t = 0, time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        cam.release()


if __name__ == "__main__":
    main()
