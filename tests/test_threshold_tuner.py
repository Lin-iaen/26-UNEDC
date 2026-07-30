#!/usr/bin/env python3
"""阈值调参工具 —— 手动固定管道ROI，实时调试球检测参数。

用法：
    source venv/bin/activate
    python tests/test_threshold_tuner.py
    → 浏览器打开 http://<pi-ip>:5002

操作：
    1. 拖动管道ROI滑块固定管道上下边界（黄框）
    2. 切换 thresh / canny 检测器并调参
    3. 点击「锁定ROI」保存到 calibration_data/ball_tracker.json
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from flask import jsonify, request

from src.drivers import Camera
from src.vision import MjpegStreamer
from src.ball_tracker import calibrate

CALIB_FILE = calibrate.CALIB_FILE

SENSOR_MODE = 1
OUTPUT_W, OUTPUT_H = 640, 480
TUNER_PORT = 5002

DEFAULT_Y1 = 200
DEFAULT_Y2 = 260

THRESH_PARAMS = {
    "blockSize": (11, 51, 2, 21),
    "C": (2, 20, 1, 4),
    "morph_size": (1, 15, 2, 5),
    "min_area": (10, 500, 5, 50),
    "max_area": (100, 5000, 100, 800),
}

CANNY_PARAMS = {
    "low": (10, 200, 5, 30),
    "high": (30, 400, 10, 100),
    "minR": (5, 50, 1, 8),
    "maxR": (20, 100, 1, 40),
}

CAM_PARAMS = {
    "ExposureTime": (100, 66000, 100, 30000),
    "AnalogueGain": (1.0, 16.0, 0.1, 3.0),
}

STATE = {
    "detector": "thresh",
    "params": {k: v[3] for k, v in THRESH_PARAMS.items()},
    "cam": {k: v[3] for k, v in CAM_PARAMS.items()},
    "roi": {"y1": 180, "y2": 260, "x1": 0, "x2": OUTPUT_W - 1},
    "detected": "-",
    "fps": 0.0,
    "dbg": {},
}


# ── 检测器 (简化版: 只找暗块, 丢掉形状约束) ──────────────────

def detect_thresh(frame, roi, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ry1, ry2 = roi["y1"], roi["y2"]
    rx1, rx2 = roi["x1"], roi["x2"]
    if rx1 < rx2 and ry1 < ry2:
        roi_gray = gray[ry1:ry2, rx1:rx2]
        ox, oy = rx1, ry1
    else:
        roi_gray = gray
        ox = oy = 0

    blurred = cv2.GaussianBlur(roi_gray, (7, 7), 1.5)

    blk = max(3, int(p["blockSize"])) | 1
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blk, float(p["C"]),
    )

    ksize = max(1, int(p["morph_size"])) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = roi_gray.shape[:2]
    min_a = float(p["min_area"])
    max_a = float(p["max_area"])
    limit = max_a if max_a < h * w * 0.3 else h * w * 0.3
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_a or area > limit:
            continue
        if area > best_area:
            best_area = area
            best_cnt = cnt

    info = f"轮廓:{len(contours)}"
    if best_cnt is not None:
        (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
        info += f"  已识别 area={best_area:.0f}"
        return closed, int(cx) + ox, int(cy) + oy, int(radius), info
    return closed, None, None, None, info


def detect_canny(frame, roi, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ry1, ry2 = roi["y1"], roi["y2"]
    rx1, rx2 = roi["x1"], roi["x2"]
    if rx1 < rx2 and ry1 < ry2:
        roi_gray = gray[ry1:ry2, rx1:rx2]
        ox, oy = rx1, ry1
    else:
        roi_gray = gray
        ox = oy = 0

    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, int(p["low"]), int(p["high"]))

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=int(p["high"]), param2=25,
        minRadius=int(p["minR"]), maxRadius=int(p["maxR"]),
    )

    if circles is not None and len(circles[0]) > 0:
        c = circles[0][0]
        return edges, int(c[0]) + ox, int(c[1]) + oy, int(c[2]), \
            f"已识别 {len(circles[0])} 个圆"
    return edges, None, None, None, \
        f"无圆 (low={int(p['low'])} high={int(p['high'])})"


DETECTORS = {"thresh": detect_thresh, "canny": detect_canny}


# ── provider ──────────────────────────────────────────────────

def make_provider(cam):
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
        roi = STATE["roi"]
        proc, cx, cy, r, info = DETECTORS[det](frame, roi, p)

        # 边界检查：圆不能超出 ROI
        x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
        if cx is not None and x1 < x2 and y1 < y2:
            if cx - r < x1 or cx + r > x2 or cy - r < y1 or cy + r > y2:
                cx = cy = r = None
                info += "  X 超出ROI"

        h, w = frame.shape[:2]

        if proc is not None:
            if proc.ndim == 2:
                proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            else:
                proc_bgr = proc
            proc_bgr = cv2.resize(proc_bgr, (w, h))
        else:
            proc_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        out_frame = frame.copy()

        # ── ROI 矩形 ─────────────────────────────────────
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(out_frame, f"ROI {x1},{y1}-{x2},{y2}",
                    (x1 + 4, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 200, 0), 1)

        # ── 球标注 ───────────────────────────────────────
        if cx is not None:
            for arr in (out_frame, proc_bgr):
                cv2.circle(arr, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(arr, (cx, cy), 3, (0, 255, 0), -1)
                cv2.line(arr, (cx - r - 4, cy), (cx + r + 4, cy),
                         (0, 255, 0), 1)
                cv2.line(arr, (cx, cy - r - 4), (cx, cy + r + 4),
                         (0, 255, 0), 1)
            for yy in range(0, h, 8):
                cv2.line(proc_bgr, (cx, yy), (cx, min(yy + 4, h - 1)),
                         (0, 200, 0), 1)
            label = f"[{det}]  ({cx},{cy}) r={r}  {info}"
            STATE["detected"] = (f"({cx},{cy}) r={r}  已识别\n"
                                 f"ROI x{roi['x1']}-x{roi['x2']} "
                                 f"y{roi['y1']}-y{roi['y2']}")
        else:
            label = f"[{det}]  {info}"
            STATE["detected"] = (f"{info}\n"
                                 f"ROI x{roi['x1']}-x{roi['x2']} "
                                 f"y{roi['y1']}-y{roi['y2']}")

        out = np.hstack((out_frame, proc_bgr))

        cv2.putText(out, "--- orig ---", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        cv2.putText(out, "--- proc ---", (w + 6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        cv2.putText(out, label, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return out

    return provider


# ── HTML ──────────────────────────────────────────────────────

def build_page():
    def _s(items, det):
        return "".join(
            f'<div class="sr"><label>{k} <span id="{det[0]}v_{k}" class="v"></span></label>'
            f'<input type="range" id="{det[0]}s_{k}" min="{v[0]}" max="{v[1]}" '
            f'step="{v[2]}" value="{v[3]}" '
            f'oninput="setP(\'{det}\',\'{k}\',this.value)"></div>'
            for k, v in items
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>阈值调参</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}}
#pn{{width:380px;overflow-y:auto;padding:12px;background:#1a1a1a;border-right:1px solid #333}}
#main{{flex:1;display:flex;align-items:center;justify-content:center;background:#000}}
img{{max-width:100%;max-height:100vh;object-fit:contain}}
h3{{margin:12px 0 6px;font-size:12px;color:#6cf;border-bottom:1px solid #333;padding-bottom:3px}}
.sr{{margin:6px 0}}
.sr label{{display:block;font-size:11px;color:#aaa;margin-bottom:2px}}
.sr input[type=range]{{width:100%}}
.v{{float:right;color:#0f0;font-size:11px}}
.btn{{display:inline-block;margin:3px 2px;padding:5px 9px;background:#2a2a2a;color:#ddd;
     border:1px solid #444;cursor:pointer;font-size:11px;font-family:monospace}}
.btn:hover{{background:#3a3a3a}}
.btn.on{{background:#1a3a1a;border-color:#2a6a2a;color:#8f8}}
#info{{font-size:11px;color:#fa0;margin-top:8px;min-height:14px}}
</style></head><body>
<div id="pn">
<h3>相机参数（AE=关闭）</h3>
<div class="sr"><label>ExposureTime μs <span id="ev_et" class="v">30000</span></label>
<input type="range" id="es_et" min="100" max="66000" step="100" value="30000"
 oninput="setCam('ExposureTime',this.value)"></div>
<div class="sr"><label>AnalogueGain x <span id="ev_ag" class="v">3.0</span></label>
<input type="range" id="es_ag" min="1.0" max="16.0" step="0.1" value="3.0"
 oninput="setCam('AnalogueGain',this.value)"></div>
<span class="btn" onclick="aeOn()">开启AE</span>
<span id="ae_msg" style="font-size:10px;color:#aaa;margin-left:6px"></span>

<h3>管道 ROI（黄框范围）</h3>
<div class="sr"><label>上边界 y1 <span id="roi_v1" class="v">180</span></label>
<input type="range" id="roi_s1" min="0" max="479" step="1" value="180"
 oninput="setROI('y1',this.value)"></div>
<div class="sr"><label>下边界 y2 <span id="roi_v2" class="v">260</span></label>
<input type="range" id="roi_s2" min="0" max="479" step="1" value="260"
 oninput="setROI('y2',this.value)"></div>
<div class="sr"><label>左边界 x1 <span id="roi_v3" class="v">0</span></label>
<input type="range" id="roi_s3" min="0" max="639" step="1" value="0"
 oninput="setROI('x1',this.value)"></div>
<div class="sr"><label>右边界 x2 <span id="roi_v4" class="v">639</span></label>
<input type="range" id="roi_s4" min="0" max="639" step="1" value="639"
 oninput="setROI('x2',this.value)"></div>
<span class="btn" onclick="lockROI()">锁定ROI</span>
<span id="roi_msg" style="font-size:10px;color:#aaa;margin-left:6px"></span>

<h3>检测器</h3>
<span class="btn on" id="b_t" onclick="sw('thresh')">阈值</span>
<span class="btn" id="b_c" onclick="sw('canny')">Canny</span>

<h3 id="ph_t">阈值参数</h3>
<div id="pan_t">{_s(THRESH_PARAMS.items(),'thresh')}</div>

<h3 id="ph_c" style="display:none">Canny参数</h3>
<div id="pan_c" style="display:none">{_s(CANNY_PARAMS.items(),'canny')}</div>

<div id="info">--</div>
</div>
<div id="main"><img src="/video_feed"></div>

<script>
var active='thresh';
function show(det){{
  ['pan_t','pan_c','ph_t','ph_c'].forEach(function(id){{
    document.getElementById(id).style.display='none'
  }})
  document.getElementById('pan_'+det[0]).style.display='block'
  document.getElementById('ph_'+det[0]).style.display='block'
  document.getElementById('b_t').className='btn'+(det==='thresh'?' on':'')
  document.getElementById('b_c').className='btn'+(det==='canny'?' on':'')
}}
var _p={{}},_t={{}};
function setP(det,k,v){{
  document.getElementById((det==='thresh'?'tv_':'cv_')+k).textContent=v
  _p[k]=v; if(_t[k])return
  _t[k]=setTimeout(function(){{_t[k]=null
    fetch('/set?'+k+'='+_p[k]).catch(e=>{{}})}},80)
}}
function sw(det){{active=det;show(det)
  fetch('/set?detector='+det).catch(e=>{{}})}}

var _cam={{}},_ct={{}};
function setCam(k,v){{
  document.getElementById('ev_'+(k==='ExposureTime'?'et':'ag')).textContent=v
  _cam[k]=v; if(_ct[k])return
  _ct[k]=setTimeout(function(){{_ct[k]=null
    fetch('/set_cam?'+k+'='+_cam[k]).catch(e=>{{}})}},60)
}}
function aeOn(){{
  fetch('/ae_on').then(r=>r.json()).then(function(d){{
    document.getElementById('ae_msg').textContent='AE已开启'
    setTimeout(function(){{document.getElementById('ae_msg').textContent=''}},3000)
  }}).catch(e=>{{}})
}}

var _r={{}};
function setROI(k,v){{
  var idx={{'y1':1,'y2':2,'x1':3,'x2':4}}[k];
  document.getElementById('roi_v'+idx).textContent=v;
  _r[k]=v;
  clearTimeout(_r.timer);
  _r.timer=setTimeout(function(){{
    fetch('/set_roi?'+k+'='+(_r[k]||document.getElementById('roi_s'+idx).value))
    .catch(e=>{{}})}},60);
}}
function lockROI(){{
  fetch('/lock_roi').then(r=>r.json()).then(function(d){{
    document.getElementById('roi_msg').textContent=d.ok?'已保存':'失败'
    setTimeout(function(){{document.getElementById('roi_msg').textContent=''}},3000)
  }}).catch(e=>{{}})
}}

function poll(){{
  fetch('/stats').then(r=>r.json()).then(function(d){{
    document.getElementById('info').textContent=d.detected+'  '+d.fps+' fps';
    ['roi_s1','roi_s2','roi_s3','roi_s4'].forEach(function(id,i){{
      var key=['y1','y2','x1','x2'][i];
      document.getElementById('roi_v'+(i+1)).textContent=d.roi[key];
      document.getElementById(id).value=d.roi[key];
    }})
    if(d.cam){{
      document.getElementById('ev_et').textContent=d.cam.ExposureTime;
      document.getElementById('es_et').value=d.cam.ExposureTime;
      document.getElementById('ev_ag').textContent=d.cam.AnalogueGain;
      document.getElementById('es_ag').value=d.cam.AnalogueGain;
    }}
    for(var k in d.params){{
      var el=document.getElementById((active==='thresh'?'tv_':'cv_')+k);
      if(el) el.textContent=d.params[k];
    }}
  }}).catch(e=>{{}})
}}

show('thresh');poll();setInterval(poll,1000)
</script></body></html>"""


# ── 路由 ──────────────────────────────────────────────────────

def make_routes(cam):
    def route_stats(**kw):
        return jsonify({
            "detected": STATE["detected"],
            "fps": STATE["fps"],
            "params": STATE["params"],
            "cam": STATE["cam"],
            "roi": STATE["roi"],
        })

    def route_set(**kw):
        for key in request.args:
            if key == "detector":
                d = request.args[key]
                if d in DETECTORS:
                    STATE["detector"] = d
                    STATE["params"] = {k: v[3] for k, v in
                                       (THRESH_PARAMS if d == "thresh"
                                        else CANNY_PARAMS).items()}
                continue
            if key in STATE["params"]:
                try:
                    STATE["params"][key] = float(request.args[key])
                except ValueError:
                    pass
        return jsonify({"ok": True})

    def route_set_cam(**kw):
        for key in request.args:
            if key in CAM_PARAMS:
                try:
                    val = float(request.args[key])
                    ctrl = {key: int(val) if key == "ExposureTime" else val}
                    cam.set_params(ctrl)
                    STATE["cam"][key] = val
                except (ValueError, Exception):
                    pass
        return jsonify({"ok": True, "cam": STATE["cam"]})

    def route_ae_on(**kw):
        cam.set_params({"AeEnable": True})
        return jsonify({"ok": True})

    def route_set_roi(**kw):
        for key in request.args:
            if key in ("y1", "y2", "x1", "x2"):
                try:
                    limit = OUTPUT_H - 1 if key.startswith("y") else OUTPUT_W - 1
                    STATE["roi"][key] = max(0, min(int(request.args[key]), limit))
                except ValueError:
                    pass
        return jsonify({"ok": True, "roi": STATE["roi"]})

    def route_lock_roi(**kw):
        if CALIB_FILE.exists():
            try:
                calib = json.loads(CALIB_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                calib = {}
        else:
            calib = {}
        for k in ("x1", "x2", "y1", "y2"):
            calib[f"pipe_roi_{k}"] = STATE["roi"][k]
        calibrate.save(calib)
        return jsonify({"ok": True, "roi": STATE["roi"]})

    return {
        "/stats": route_stats,
        "/set": route_set,
        "/set_cam": route_set_cam,
        "/ae_on": route_ae_on,
        "/set_roi": route_set_roi,
        "/lock_roi": route_lock_roi,
    }


# ── 入口 ──────────────────────────────────────────────────────

def main():
    cam = Camera(vflip=True, hflip=True, output_size=(OUTPUT_W, OUTPUT_H))
    cam.start()

    # load existing ROI from calibration if available
    try:
        calib = calibrate.load()
        if calib:
            if "pipe_roi_y1" in calib and "pipe_roi_y2" in calib:
                STATE["roi"]["y1"] = int(calib["pipe_roi_y1"])
                STATE["roi"]["y2"] = int(calib["pipe_roi_y2"])
            if "pipe_roi_x1" in calib and "pipe_roi_x2" in calib:
                STATE["roi"]["x1"] = int(calib["pipe_roi_x1"])
                STATE["roi"]["x2"] = int(calib["pipe_roi_x2"])
            print(f"ROI loaded: x{STATE['roi']['x1']}-x{STATE['roi']['x2']} "
                  f"y{STATE['roi']['y1']}-y{STATE['roi']['y2']}")
    except Exception:
        pass

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
    cam.set_params({
        "AeEnable": False,
        "ExposureTime": int(STATE["cam"]["ExposureTime"]),
        "AnalogueGain": STATE["cam"]["AnalogueGain"],
    })

    streamer = MjpegStreamer(
        frame_provider=make_provider(cam),
        port=TUNER_PORT, max_fps=30.0,
        custom_template=build_page(),
        custom_routes=make_routes(cam),
    )
    streamer.start()
    print(f"调参工具 → http://<pi-ip>:{TUNER_PORT}")

    fps_c, fps_t = 0, time.perf_counter()
    try:
        while True:
            time.sleep(0.05)
            fps_c += 1
            if time.perf_counter() - fps_t >= 1.0:
                STATE["fps"] = round(fps_c / (time.perf_counter() - fps_t), 1)
                fps_c, fps_t = 0, time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        cam.release()


if __name__ == "__main__":
    main()
