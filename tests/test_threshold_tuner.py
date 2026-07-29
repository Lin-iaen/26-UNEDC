#!/usr/bin/env python3
"""阈值调参工具 —— 实时调节检测参数，边调边看效果。

用法：
    source venv/bin/activate
    python tests/test_threshold_tuner.py
    → 浏览器打开 http://<pi-ip>:5002

支持两种检测器：
  自适应阈值 (thresh) — blockSize, C, morph_size, min_area, max_area, circularity
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
from src.ball_tracker import pipe_detector

SENSOR_MODE = 1
OUTPUT_W, OUTPUT_H = 640, 480
TUNER_PORT = 5002
IMG_AREA = OUTPUT_W * OUTPUT_H

THRESH_PARAMS = {
    "blockSize": (11, 51, 2, 21),
    "C": (2, 20, 1, 4),
    "morph_size": (1, 15, 2, 5),
    "min_area": (10, 500, 10, 50),
    "max_area": (500, 150000, 1000, 90000),
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
    "dbg": {},
    "pipe": None,
}


def apply_thresh(frame, p):
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
    max_a = float(p["max_area"])
    circ_th = float(p["circularity"])
    best = None
    best_circ = 0.0
    best_cnt = None
    best_area_val = 0
    reject_reason = ""

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_a:
            reject_reason = f"area={area:.0f}<{min_a:.0f}"
            continue
        if area > max_a:
            reject_reason = f"area={area:.0f}>{max_a:.0f}"
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            reject_reason = "peri=0"
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < circ_th:
            reject_reason = f"circ={circularity:.2f}<{circ_th:.2f}"
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        ca = np.pi * radius * radius
        fr = area / ca if ca > 0 else 0
        if fr < 0.3:
            reject_reason = f"fill={fr:.2f}<0.3"
            continue
        if circularity > best_circ:
            best_circ = circularity
            best = (int(cx), int(cy), int(radius))
            best_cnt = cnt
            best_area_val = int(area)
            reject_reason = ""

    info = f"轮廓:{len(contours)}"
    if best:
        info += f"  已识别 area={best_area_val} circ={best_circ:.2f}"
    elif reject_reason:
        info += f"  筛除:{reject_reason}"
    else:
        info += "  无轮廓"

    STATE["dbg"] = {
        "n": len(contours),
        "best": best,
        "best_cnt": best_cnt,
        "best_area": best_area_val,
        "best_circ": best_circ,
        "reject": reject_reason,
    }

    return closed, *(best or (None, None, None)), info


def apply_canny(frame, p):
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
        return result, int(c[0]), int(c[1]), int(c[2]), \
            f"已识别 {len(circles[0])} 个圆"
    return result, None, None, None, \
        f"未检测到圆 (Canny low={int(p['low'])} high={int(p['high'])})"


DETECTORS = {"thresh": apply_thresh, "canny": apply_canny}
PARAM_SETS = {"thresh": THRESH_PARAMS, "canny": CANNY_PARAMS}


def make_frame_provider(cam):
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

        # ── 管道检测 ────────────────────────────────────────────────
        pipe = pipe_detector.detect(frame)
        STATE["pipe"] = (
            f"pipe W={pipe['width_px']}px @{pipe['axis_angle']:.0f}deg" if pipe
            else "未检测到管道"
        )

        # ── Sobel 边缘投影可视化 ──────────────────────────
        gray_for_sobel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray_for_sobel, cv2.CV_32F, 0, 1, ksize=3)
        abs_sobel = np.abs(sobel_y)
        prof = np.sum(abs_sobel, axis=1).astype(np.float32)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (15, 1), 3).flatten()
        proj_vals = prof / (np.max(prof) + 1e-6)

        det = STATE["detector"]
        p = STATE["params"]
        proc, cx, cy, r, debug_info = DETECTORS[det](frame, p)
        dbg = STATE.get("dbg", {})

        # ── 管道 ROI 后处理：排除管道外的误检 ────────
        if cx is not None and pipe is not None:
            mask = pipe["roi_mask"]
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                if mask[cy, cx] == 0:
                    cx = cy = r = None
                    debug_info += "  X pipe"
            else:
                cx = cy = r = None
                debug_info += "  X OOB"

        h, w = frame.shape[:2]

        if proc is not None:
            if proc.ndim == 2:
                proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            else:
                proc_bgr = proc
            proc_bgr = cv2.resize(proc_bgr, (w, h))
        else:
            proc_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        # ── 管道 overlay（无论成败都画，None 时画 "NO PIPE"）─
        frame = pipe_detector.draw_overlay(frame, pipe)

        # ── Sobel 投影条 + 标签 ───────────────────────
        bar_w = min(36, w // 10)
        for yi in range(h):
            lw = int(proj_vals[yi] * (bar_w - 1))
            if lw >= 1:
                cv2.line(proc_bgr, (w - bar_w, yi),
                         (w - bar_w + lw, yi), (60, 120, 255), 1)
        cv2.line(proc_bgr, (w - bar_w, 0), (w - bar_w, h - 1), (60, 60, 60), 1)
        cv2.putText(proc_bgr, "Sobel", (w - bar_w - 36, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 180), 1)
        if pipe is not None:
            for yp in (pipe["y_top"], pipe["y_bottom"]):
                cv2.line(proc_bgr, (w - bar_w - 4, yp),
                         (w - 1, yp), (255, 200, 0), 1)

        # ── 球检测绘图 ──────────────────────────────────
        if cx is not None:
            best_cnt = dbg.get("best_cnt")
            if best_cnt is not None:
                cv2.drawContours(frame, [best_cnt], -1, (0, 180, 255), 2)

            for arr in (frame, proc_bgr):
                cv2.circle(arr, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(arr, (cx, cy), 3, (0, 255, 0), -1)
                cv2.line(arr, (cx - r - 4, cy), (cx + r + 4, cy),
                         (0, 255, 0), 1)
                cv2.line(arr, (cx, cy - r - 4), (cx, cy + r + 4),
                         (0, 255, 0), 1)

            for yy in range(0, h, 8):
                cv2.line(proc_bgr, (cx, yy), (cx, min(yy + 4, h - 1)),
                         (0, 200, 0), 1)

        # ── 拼合 ──────────────────────────────────────────
        out = np.hstack((frame, proc_bgr))

        cv2.putText(out, "--- orig ---", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        cv2.putText(out, "--- proc ---", (w + 6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        # ── status dot (top center) ────────────────────
        status_color = (0, 255, 0) if cx is not None else (
            (0, 140, 255) if pipe else (0, 0, 255))
        cv2.circle(out, (out.shape[1] // 2, 12), 6, status_color, -1)
        cv2.circle(out, (out.shape[1] // 2, 12), 6, (255, 255, 255), 1)

        if cx is not None:
            area_val = dbg.get("best_area", 0)
            circ_val = dbg.get("best_circ", 0)
            label = f"[{det}]  ({cx},{cy})  r={r}px  "
            label += f"area={area_val}  circ={circ_val:.2f}"
            STATE["detected"] = (
                f"({cx},{cy}) r={r} a={area_val} c={circ_val:.2f}  已识别"
                f"  |  {STATE['pipe']}")
        else:
            tip = debug_info
            label = f"[{det}]  {tip}"
            STATE["detected"] = f"{tip}  |  {STATE['pipe']}"

        cv2.putText(out, label, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return out

    return provider


# ── HTML + JS ────────────────────────────────────────────────────────

def build_page():
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
    fetch('/set?'+key+'='+pending[key]).catch(e=>{{}});
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

    fps_c, fps_t = 0, time.perf_counter()

    try:
        while True:
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
