#!/usr/bin/env python3
"""
全参数调试 + 追踪误差测试
=========================

覆盖硬件参数（曝光/增益/亮度/对比度/饱和度/锐度/EV）、检测过滤参数、
二值化参数、追踪参数。支持预设保存/加载。

用法：
    python tests/test_tracking_test.py
    → 打开 http://<pi-ip>:5000
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.drivers import Camera
from src.vision import MjpegStreamer

# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_PARAMS = {  # 会通过 cam._cam.capture_metadata() 同步到硬件
    "ExposureTime": 20000,
    "AnalogueGain": 1.0,
    "Brightness":    0.0,
    "Contrast":      1.0,
    "Saturation":    1.0,
    "Sharpness":     1.0,
    "ExposureValue": 0.0,
}
DETECT_PARAMS = {   # 纯软件参数，不接触硬件
    "MinArea":        500,
    "MaxAspectRatio": 3.0,
    "MinContrast":    30,
}
BINARY_PARAMS = {
    "GlobalThresh": 120,
}
TRACK_PARAMS = {
    "TrackingThreshold": 20,
}
PARAMS = {**CAMERA_PARAMS, **DETECT_PARAMS, **BINARY_PARAMS, **TRACK_PARAMS}

PRESETS_DIR = Path(__file__).resolve().parent.parent / "calibration_data"

CLR_OUTER   = (0, 255, 0)
CLR_INNER   = (255, 0, 0)
CLR_DIAG    = (0, 0, 255)
CLR_CENTER  = (0, 255, 255)
CLR_RECT    = (255, 255, 0)
CLR_LINE    = (255, 255, 255)
CLR_REJECT  = (80, 80, 80)
CLR_REJ_TXT = (200, 200, 200)
CLR_OK      = (0, 255, 0)
CLR_FAIL    = (0, 0, 255)

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Tracking</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:320px;overflow-y:auto;padding:10px;background:#1a1a1a;border-right:1px solid #333}
#main{flex:1;display:flex;align-items:center;justify-content:center;padding:4px}
img{max-width:100%;max-height:100%}
h3{margin:10px 0 6px;font-size:12px;border-bottom:1px solid #333;padding-bottom:3px}
.stat{display:flex;justify-content:space-between;margin:2px 0;font-size:11px}
.stat .l{color:#aaa}.stat .v{color:#0f0;font-weight:bold}
label{display:block;margin:6px 0 2px;font-size:11px;color:#aaa}
input[type=range]{width:100%}
.btn{display:inline-block;margin:2px;padding:3px 8px;background:#333;color:#fff;border:1px solid #555;cursor:pointer;font-size:10px;text-decoration:none}
.btn:hover{background:#555}
.preset-list{font-size:10px;color:#666;margin:4px 0}
.preset-link{color:#0f0;cursor:pointer;margin:0 4px}
input[type=text]{background:#333;border:1px solid #555;color:#fff;padding:2px 4px;font-size:10px;width:100px}
</style></head><body>
<div id="panel">
<div id="js_status" style="color:#f80;font-size:10px;margin-bottom:4px">JS loading...</div>
<div class="stat"><span class="l">矩形</span><span class="v" id="s_rects">--</span></div>
<div class="stat"><span class="l">FPS</span><span class="v" id="stat_fps">--</span></div>

<h3>📷 相机参数</h3>
<label>曝光 <span class="val" id="v_ExposureTime">--</span></label>
<input type="range" id="sl_ExposureTime" min="39" max="66666" oninput="setParam('ExposureTime',parseInt(this.value))">
<label>增益 <span class="val" id="v_AnalogueGain">--</span></label>
<input type="range" id="sl_AnalogueGain" min="1" max="16" step="0.1" oninput="setParam('AnalogueGain',parseFloat(this.value))">
<label>亮度 <span class="val" id="v_Brightness">--</span></label>
<input type="range" id="sl_Brightness" min="-1" max="1" step="0.05" oninput="setParam('Brightness',parseFloat(this.value))">
<label>对比度 <span class="val" id="v_Contrast">--</span></label>
<input type="range" id="sl_Contrast" min="0" max="32" step="0.1" oninput="setParam('Contrast',parseFloat(this.value))">
<label>饱和度 <span class="val" id="v_Saturation">--</span></label>
<input type="range" id="sl_Saturation" min="0" max="32" step="0.1" oninput="setParam('Saturation',parseFloat(this.value))">
<label>锐度 <span class="val" id="v_Sharpness">--</span></label>
<input type="range" id="sl_Sharpness" min="0" max="16" step="0.1" oninput="setParam('Sharpness',parseFloat(this.value))">
<label>EV补偿 <span class="val" id="v_ExposureValue">--</span></label>
<input type="range" id="sl_ExposureValue" min="-4" max="4" step="0.5" oninput="setParam('ExposureValue',parseFloat(this.value))">
<a class="btn" href="/reset_ae">↻ 恢复自动曝光</a>

<h3>🔍 检测</h3>
<label>最小面积 <span class="val" id="v_MinArea">--</span></label>
<input type="range" id="sl_MinArea" min="100" max="10000" oninput="setParam('MinArea',parseInt(this.value))">
<label>最大长宽比 <span class="val" id="v_MaxAspectRatio">--</span></label>
<input type="range" id="sl_MaxAspectRatio" min="1" max="10" step="0.1" oninput="setParam('MaxAspectRatio',parseFloat(this.value))">
<label>最小对比度 <span class="val" id="v_MinContrast">--</span></label>
<input type="range" id="sl_MinContrast" min="10" max="100" oninput="setParam('MinContrast',parseInt(this.value))">

<h3>🧠 二值化</h3>
<label>阈值 <span class="val" id="v_GlobalThresh">--</span></label>
<input type="range" id="sl_GlobalThresh" min="0" max="255" oninput="setParam('GlobalThresh',parseInt(this.value))">

<h3>🎯 追踪</h3>
<label>追踪阈值 <span class="val" id="v_TrackingThreshold">--</span> px</label>
<input type="range" id="sl_TrackingThreshold" min="5" max="100" oninput="setParam('TrackingThreshold',parseInt(this.value))">

<h3>💾 预设</h3>
<input type="text" id="preset_name" placeholder="预设名">
<button class="btn" onclick="savePreset()">保存</button>
<button class="btn" onclick="loadPreset()">加载</button>
<div class="preset-list">已有: <span id="preset_list"></span></div>

</div>
<div id="main">
<img src="/video_feed" id="stream">
</div>
<script>
const ALL_KEYS = [
  "ExposureTime","AnalogueGain","Brightness","Contrast","Saturation","Sharpness",
  "ExposureValue","MinArea","MaxAspectRatio","MinContrast",
  "GlobalThresh","TrackingThreshold"
];
const ALL_SLIDERS = [1,1,1,1,1,1,1,1,1,1,1,1]; // all have sliders

function toKey(k){return k.charAt(0).toLowerCase()+k.slice(1);}
function setParam(name,v){
  document.getElementById('v_'+name).textContent=v;
  document.getElementById('sl_'+name).value=v;
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/set?'+toKey(name)+'='+v, false);
  xhr.send();
}
function poll(){
  document.getElementById('js_status').textContent = 'JS polling...';
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/stats?_='+Date.now(), false);
  try {
    xhr.send();
    var d = JSON.parse(xhr.responseText);
    document.getElementById('js_status').textContent = 'OK: '+d.rects+'r '+d.fps+'fps';
    ALL_KEYS.forEach(function(k){
      var jk=toKey(k); var v=d[jk];
      if(v!==undefined && v!==null){
        var elV = document.getElementById('v_'+k);
        var elS = document.getElementById('sl_'+k);
        if(elV) elV.textContent = v;
        if(elS) elS.value = v;
      }
    });
    var elFps = document.getElementById('stat_fps');
    var elR = document.getElementById('s_rects');
    if(elFps) elFps.textContent = d.fps;
    if(elR) elR.textContent = d.rects;
  } catch(e) {
    document.getElementById('js_status').textContent = 'ERR: '+e;
  }
}
function refreshPresets(){
  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function(){
    if(xhr.readyState==4 && xhr.status==200){
      var d = JSON.parse(xhr.responseText);
      var html = '';
      d.presets.forEach(function(p){
        html += '<span class=\"preset-link\" onclick=\"loadPreset(\\''+p+'\\')\">'+p+'</span>';
      });
      document.getElementById('preset_list').innerHTML = html || '--';
    }
  };
  xhr.open('GET', '/presets', true);
  xhr.send();
}
function savePreset(){
  var name = document.getElementById('preset_name').value || 'my_preset';
  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function(){
    if(xhr.readyState==4 && xhr.status==200){
      var d = JSON.parse(xhr.responseText);
      if(d.ok) refreshPresets();
    }
  };
  xhr.open('GET', '/save?name='+name, true);
  xhr.send();
}
function loadPreset(name){
  var n = name || document.getElementById('preset_name').value || 'default';
  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function(){
    if(xhr.readyState==4 && xhr.status==200){
      var d = JSON.parse(xhr.responseText);
      if(d.ok){ poll(); refreshPresets(); }
    }
  };
  xhr.open('GET', '/load?name='+n, true);
  xhr.send();
}

poll();
refreshPresets();
setInterval(poll, 2000);
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 检测辅助（与 detect 版一致）
# ══════════════════════════════════════════════════════════════════════════════

def _is_rectangle(approx: np.ndarray) -> bool:
    return len(approx) == 4 and cv2.isContourConvex(approx)

def _aspect_ratio(cnt: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(cnt)
    if min(w, h) == 0: return 999.0
    return max(w / h, h / w)

def _contrast_check(gray, outer_approx, inner_approx):
    x, y, w, h = cv2.boundingRect(outer_approx)
    roi = gray[y:y+h, x:x+w]
    oa = outer_approx.copy(); ia = inner_approx.copy()
    oa[:,0,0] -= x; oa[:,0,1] -= y
    ia[:,0,0] -= x; ia[:,0,1] -= y
    mask_in = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask_in, [ia], -1, 255, cv2.FILLED)
    mask_out = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask_out, [oa], -1, 255, cv2.FILLED)
    tape = cv2.subtract(mask_out, mask_in)
    return cv2.mean(roi, mask=mask_in)[0] - cv2.mean(roi, mask=tape)[0]

def _order_corners(approx):
    pts = approx.reshape(4, 2)
    idx = np.lexsort((pts[:,0], pts[:,1]))
    pts = pts[idx]
    top = pts[:2][np.argsort(pts[:2,0])]
    bottom = pts[2:][np.argsort(pts[2:,0])[::-1]]
    return np.vstack([top, bottom])

def _center_of(ordered: np.ndarray) -> tuple[int, int]:
    cx = (ordered[0,0] + ordered[1,0] + ordered[2,0] + ordered[3,0]) // 4
    cy = (ordered[0,1] + ordered[1,1] + ordered[2,1] + ordered[3,1]) // 4
    return int(cx), int(cy)

def _draw_crosshair(img, cx, cy, size=10, color=(0,255,255), thickness=1):
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(img, (cx, cy), 2, color, -1)

# ══════════════════════════════════════════════════════════════════════════════
# 处理
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(frame: np.ndarray, params: dict) -> tuple[np.ndarray, int, dict]:
    h, w = frame.shape[:2]
    input_w, input_h = w, h
    MAX_W, MAX_H = 640, 480
    if w > MAX_W or h > MAX_H:
        scale = min(MAX_W / w, MAX_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        h, w = new_h, new_w
    if _frame_count == 0:
        print(f"Camera output: {input_w}x{input_h} → processing: {w}x{h}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_area   = int(params.get("MinArea", 500))
    max_ar     = float(params.get("MaxAspectRatio", 3.0))
    min_contrast = int(params.get("MinContrast", 30))
    global_thresh = int(params.get("GlobalThresh", 120))
    track_thresh  = int(params.get("TrackingThreshold", 20))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)

    debug_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    detection_view = frame.copy()
    frame_cx, frame_cy = w // 2, h // 2
    _draw_crosshair(detection_view, frame_cx, frame_cy, 12, CLR_CENTER, 2)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_view, contours, -1, (70, 70, 70), 1)

    error = {"dx": 0, "dy": 0, "distance": 0.0, "tracking": False,
             "rect_center_x": 0, "rect_center_y": 0}

    if hierarchy is None:
        return _compose_views(detection_view, debug_view, 0, error), 0, error

    hierarchy = hierarchy[0]
    found_pairs = []
    rejected = []

    for idx, (cnt, hinfo) in enumerate(zip(contours, hierarchy)):
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if not _is_rectangle(approx): continue
        ar = _aspect_ratio(approx)
        if ar > max_ar:
            rejected.append((approx, f"AR={ar:.1f}"))
            continue
        child_idx = hinfo[2]
        if child_idx == -1: continue
        child_cnt = contours[child_idx]
        child_epsilon = 0.02 * cv2.arcLength(child_cnt, True)
        child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
        if not _is_rectangle(child_approx): continue
        contrast = _contrast_check(gray, approx, child_approx)
        if contrast <= min_contrast:
            rejected.append((approx, f"C={contrast:.0f}"))
            continue
        found_pairs.append((approx, child_approx))

    for cnt, reason in rejected:
        cv2.drawContours(detection_view, [cnt], -1, CLR_REJECT, 2)
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        cv2.putText(detection_view, reason, (rx + rw + 4, ry + rh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_REJ_TXT, 1, cv2.LINE_AA)

    for outer, inner in found_pairs:
        cv2.drawContours(detection_view, [outer], -1, CLR_OUTER, 2)
        cv2.drawContours(detection_view, [inner], -1, CLR_INNER, 2)
        ordered = _order_corners(inner)
        cv2.line(detection_view, tuple(ordered[0]), tuple(ordered[2]), CLR_DIAG, 2)
        cv2.line(detection_view, tuple(ordered[1]), tuple(ordered[3]), CLR_DIAG, 2)
        rect_cx, rect_cy = _center_of(ordered)
        _draw_crosshair(detection_view, rect_cx, rect_cy, 10, CLR_RECT, 2)
        cv2.line(detection_view, (frame_cx, frame_cy), (rect_cx, rect_cy), CLR_LINE, 1)
        dx = rect_cx - frame_cx
        dy = rect_cy - frame_cy
        dist = np.sqrt(dx**2 + dy**2)
        tracking = bool(dist <= track_thresh)
        error = {"dx": dx, "dy": dy, "distance": round(dist, 1),
                 "tracking": tracking, "rect_center_x": rect_cx, "rect_center_y": rect_cy}
        mid_x = (frame_cx + rect_cx) // 2
        mid_y = (frame_cy + rect_cy) // 2
        if tracking:
            cv2.putText(detection_view, "TRACK ✓", (mid_x - 30, mid_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_OK, 2, cv2.LINE_AA)
        else:
            cv2.putText(detection_view, f"dx:{dx:+d} dy:{dy:+d}", (mid_x - 30, mid_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_LINE, 1, cv2.LINE_AA)

    return _compose_views(detection_view, debug_view, len(found_pairs), error), len(found_pairs), error


def _compose_views(detection, debug, rect_count, error) -> np.ndarray:
    h, w = detection.shape[:2]
    pip_w, pip_h = w // 4, h // 4
    debug_small = cv2.resize(debug, (pip_w, pip_h))
    result = detection.copy()
    rx, ry = w - pip_w - 8, h - pip_h - 8
    result[ry:ry + pip_h, rx:rx + pip_w] = debug_small
    cv2.rectangle(result, (rx - 1, ry - 1), (rx + pip_w, ry + pip_h), (0, 255, 0), 1)
    label = "BINARY"
    cv2.putText(result, label, (rx + 3, ry - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    if error["tracking"]:
        text = f"TRACK ✓ d:{error['distance']}"
        colour = CLR_OK
    elif rect_count > 0:
        text = f"dx:{error['dx']:+d} dy:{error['dy']:+d} d:{error['distance']}"
        colour = CLR_FAIL
    else:
        text = f"R:{rect_count}"
        colour = (0, 0, 255)
    cv2.putText(result, text, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════════════════════════════════

def make_route_stats():
    from flask import jsonify
    def handler(**kwargs):
        return jsonify({
            "fps": round(_fps, 1),
            "rects": _rect_count,
            "time": datetime.now().strftime("%H:%M:%S"),
            "dx": _latest_error["dx"],
            "dy": _latest_error["dy"],
            "distance": _latest_error["distance"],
            "tracking": _latest_error["tracking"],
            **{k[0].lower() + k[1:]: v for k, v in PARAMS.items()},
        })
    return handler


def make_route_set(cam: Camera):
    from flask import jsonify, request
    def handler(**kwargs):
        for key in request.args:
            raw = request.args[key]
            try:
                val = float(raw)
                val = int(val) if val == int(val) and "." not in raw else val
            except ValueError:
                continue
            matched = next((pk for pk in PARAMS if pk.lower() == key.lower()), None)
            if matched:
                PARAMS[matched] = val
                if matched in CAMERA_PARAMS:
                    cam.set_params({matched: val})
        return jsonify({"ok": True})
    return handler


def make_route_reset_ae(cam: Camera):
    from flask import jsonify
    def handler(**kwargs):
        cam.set_params({"AeEnable": True})
        try:
            md = cam._cam.capture_metadata()
            PARAMS["ExposureTime"] = md.get("ExposureTime", 20000)
            PARAMS["AnalogueGain"] = md.get("AnalogueGain", 1.0)
        except Exception:
            pass
        print("AE re-enabled")
        return jsonify({"ok": True})
    return handler


def make_route_save():
    from flask import jsonify, request
    def handler(**kwargs):
        name = request.args.get("name", "my_preset")
        path = PRESETS_DIR / f"{name}.json"
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(PARAMS)
        payload["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Preset saved: {path}")
        return jsonify({"ok": True, "path": str(path)})
    return handler


def make_route_load(cam: Camera):
    from flask import jsonify, request
    def handler(**kwargs):
        name = request.args.get("name", "default")
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            return jsonify({"ok": False, "error": f"{name}.json not found"}), 404
        data = json.loads(path.read_text())
        for k, v in data.items():
            if k.startswith("_"): continue
            if k in PARAMS:
                PARAMS[k] = v
                if k in CAMERA_PARAMS:
                    cam.set_params({k: v})
        print(f"Preset loaded: {path}")
        return jsonify({"ok": True, "name": name})
    return handler


def make_route_presets():
    from flask import jsonify
    def handler(**kwargs):
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(p.stem for p in PRESETS_DIR.glob("*.json"))
        return jsonify({"presets": files})
    return handler


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

_frame_count = 0
_fps = 0.0
_last_ts = time.perf_counter()
_latest_error = {"dx": 0, "dy": 0, "distance": 0.0, "tracking": False,
                 "rect_center_x": 0, "rect_center_y": 0}
_rect_count = 0


def main() -> None:
    global _frame_count, _fps, _last_ts, _latest_error, _rect_count

    cam = Camera(vflip=True, sensor_size=(1280, 720))
    cam.start()

    time.sleep(1.5)
    try:
        md = cam._cam.capture_metadata()
        PARAMS["ExposureTime"] = md.get("ExposureTime", 20000)
        PARAMS["AnalogueGain"] = md.get("AnalogueGain", 1.0)
        print(f"AE locked: Expo={PARAMS['ExposureTime']}us, "
              f"Gain={PARAMS['AnalogueGain']:.2f}x")
    except Exception:
        print("Warning: could not read AE metadata")

    def frame_provider() -> np.ndarray | None:
        global _frame_count, _fps, _last_ts, _latest_error, _rect_count
        frame = cam.read()
        if frame is None:
            return None
        try:
            result, _rect_count, _latest_error = process_frame(frame, PARAMS)
        except Exception:
            import traceback; traceback.print_exc()
            result = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(result, "ERROR", (200, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _rect_count = -1
        _frame_count += 1
        now = time.perf_counter()
        if now - _last_ts >= 1.0:
            _fps = _frame_count / (now - _last_ts)
            _frame_count = 0; _last_ts = now
        return result

    streamer = MjpegStreamer(
        frame_provider=frame_provider,
        port=5000,
        custom_template=HTML_PAGE,
        custom_routes={
            "/stats":    make_route_stats(),
            "/set":      make_route_set(cam),
            "/reset_ae": make_route_reset_ae(cam),
            "/save":     make_route_save(),
            "/load":     make_route_load(cam),
            "/presets":  make_route_presets(),
        },
    )
    streamer.start()
    print("Tracking test ready at http://0.0.0.0:5000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down ...")
    streamer.stop()
    cam.release()


if __name__ == "__main__":
    main()
