#!/usr/bin/env python3
"""
自动曝光 (AE) 网页调试工具
==========================

通过网页实时观察 Picamera2 AE 的收敛过程，测试 ExposureValue 补偿效果，
以及"开启/锁死 AE"的手动/自动切换逻辑。

用法：
    python tests/test_ae_debug.py
    → 打开 http://<pi-ip>:5000
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drivers import Camera
from src.vision import MjpegStreamer

# ══════════════════════════════════════════════════════════════════════════════
# 全局相机运行时状态 —— 每帧由 frame_provider 从 metadata 更新
# ══════════════════════════════════════════════════════════════════════════════

AE_STATE = {
    "AeLocked": False,
    "ExposureTime": 0,
    "AnalogueGain": 0.0,
    "ExposureValue": 0.0,
}

_frame_count = 0
_fps = 0.0
_last_ts = time.perf_counter()

# ══════════════════════════════════════════════════════════════════════════════
# Web UI
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AE Debug</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:320px;padding:12px;background:#1a1a1a;border-right:1px solid #333}
#main{flex:1;display:flex;align-items:center;justify-content:center;padding:4px}
img{max-width:100%;max-height:100%}
h3{margin:10px 0 6px;font-size:12px;border-bottom:1px solid #333;padding-bottom:3px}
.stat{display:flex;justify-content:space-between;margin:3px 0;font-size:12px}
.stat .label{color:#aaa}.stat .val{color:#0f0;font-weight:bold}
label{display:block;margin:6px 0 2px;font-size:11px;color:#aaa}
input[type=range]{width:100%}
.btn{display:block;width:100%;margin:4px 0;padding:8px;background:#333;color:#fff;
     border:1px solid #555;cursor:pointer;font-size:12px;text-align:center}
.btn:hover{background:#555}
.btn.ae{background:#1a3a1a;border-color:#2a5a2a}
.btn.lock{background:#3a1a1a;border-color:#5a2a2a}
.val{font-size:10px;color:#0f0;float:right}
</style></head><body>
<div id="panel">
<h3>📷 AE 状态</h3>
<div class="stat"><span class="label">AeLocked</span>
  <span class="val" id="ae_locked">--</span></div>
<div class="stat"><span class="label">ExposureTime</span>
  <span class="val" id="ae_exptime">--</span></div>
<div class="stat"><span class="label">AnalogueGain</span>
  <span class="val" id="ae_gain">--</span></div>
<div class="stat"><span class="label">ExposureValue</span>
  <span class="val" id="ae_ev">--</span></div>
<div class="stat"><span class="label">FPS</span>
  <span class="val" id="stat_fps">--</span></div>

<h3>⚙️ ExposureValue</h3>
<label>EV 补偿 <span class="val" id="v_EV" style="float:right">0.0</span></label>
<input type="range" id="sl_EV" min="-4" max="4" step="0.5" value="0"
       oninput="setEV(this.value)">

<h3>🔧 控制</h3>
<button class="btn ae" onclick="enableAE()">开启 AE</button>
<button class="btn lock" onclick="lockAE()">锁死 AE</button>
</div>
<div id="main">
<img src="/video_feed" id="stream">
</div>
<script>
function setEV(v){
  document.getElementById('v_EV').textContent=v;
  fetch('/set_ev?value='+v);
}
function enableAE(){
  fetch('/enable_ae').then(r=>r.json()).then(d=>{if(d.ok) updateAll(d);});
}
function lockAE(){
  fetch('/lock_ae').then(r=>r.json()).then(d=>{if(d.ok) updateAll(d);});
}
function updateAll(d){
  document.getElementById('ae_locked').textContent = d.aeLocked ? '🔒 True' : '🔓 False';
  document.getElementById('ae_exptime').textContent = d.exposureTime + ' µs';
  document.getElementById('ae_gain').textContent = d.analogueGain.toFixed(2) + ' ×';
  document.getElementById('ae_ev').textContent = d.exposureValue.toFixed(1);
  document.getElementById('stat_fps').textContent = d.fps.toFixed(1);
  document.getElementById('v_EV').textContent = d.exposureValue.toFixed(1);
}
function poll(){
  fetch('/stats').then(r=>r.json()).then(updateAll);
}
poll();
setInterval(poll, 1000);
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 帧提供者 —— 每帧读 metadata 更新 AE_STATE
# ══════════════════════════════════════════════════════════════════════════════

def make_frame_provider(cam: Camera):
    def provider():
        global _frame_count, _fps, _last_ts
        frame = cam.read()
        if frame is None:
            return None

        # 从硬件读当前实际值 —— 不做任何视觉处理
        try:
            md = cam._cam.capture_metadata()
            AE_STATE["AeLocked"] = not md.get("AeState", 2) != 2  # AeState=2=收敛
            AE_STATE["ExposureTime"] = md.get("ExposureTime", 0)
            AE_STATE["AnalogueGain"] = md.get("AnalogueGain", 0.0)
        except Exception:
            pass

        _frame_count += 1
        now = time.perf_counter()
        if now - _last_ts >= 1.0:
            _fps = _frame_count / (now - _last_ts)
            _frame_count = 0
            _last_ts = now

        return frame
    return provider


# ══════════════════════════════════════════════════════════════════════════════
# 路由工厂
# ══════════════════════════════════════════════════════════════════════════════

def make_route_stats():
    from flask import jsonify
    def handler(**kwargs):
        return jsonify({
            "fps": round(_fps, 1),
            "aeLocked": AE_STATE["AeLocked"],
            "exposureTime": AE_STATE["ExposureTime"],
            "analogueGain": AE_STATE["AnalogueGain"],
            "exposureValue": AE_STATE["ExposureValue"],
        })
    return handler


def make_route_set_ev(cam: Camera):
    from flask import jsonify, request
    def handler(**kwargs):
        try:
            ev = float(request.args.get("value", 0))
            cam.set_params({"ExposureValue": ev})
            AE_STATE["ExposureValue"] = ev
        except Exception:
            pass
        return jsonify({"ok": True})
    return handler


def make_route_enable_ae(cam: Camera):
    from flask import jsonify
    def handler(**kwargs):
        # 开启 AE，让 ISP 重新接管曝光/增益
        cam.set_params({"AeEnable": True})
        # 立即回读以更新前端
        try:
            md = cam._cam.capture_metadata()
            AE_STATE["AeLocked"] = False
            AE_STATE["ExposureTime"] = md.get("ExposureTime", 0)
            AE_STATE["AnalogueGain"] = md.get("AnalogueGain", 0.0)
        except Exception:
            pass
        return jsonify({
            "ok": True,
            "aeLocked": AE_STATE["AeLocked"],
            "exposureTime": AE_STATE["ExposureTime"],
            "analogueGain": AE_STATE["AnalogueGain"],
        })
    return handler


def make_route_lock_ae(cam: Camera):
    from flask import jsonify
    def handler(**kwargs):
        # 读取当前瞬间的实际曝光/增益，立即锁死
        try:
            md = cam._cam.capture_metadata()
            exp = md.get("ExposureTime", 20000)
            gain = md.get("AnalogueGain", 1.0)
            cam.set_params({
                "AeEnable": False,
                "ExposureTime": exp,
                "AnalogueGain": gain,
            })
            AE_STATE["AeLocked"] = True
            AE_STATE["ExposureTime"] = exp
            AE_STATE["AnalogueGain"] = gain
        except Exception:
            pass
        return jsonify({
            "ok": True,
            "aeLocked": AE_STATE["AeLocked"],
            "exposureTime": AE_STATE["ExposureTime"],
            "analogueGain": AE_STATE["AnalogueGain"],
        })
    return handler


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cam = Camera(vflip=True)
    cam.start()

    # 等待 AE 稳定，读初始值
    time.sleep(1.5)
    try:
        md = cam._cam.capture_metadata()
        AE_STATE["ExposureTime"] = md.get("ExposureTime", 0)
        AE_STATE["AnalogueGain"] = md.get("AnalogueGain", 0.0)
        AE_STATE["AeLocked"] = not (md.get("AeState", 2) != 2)
        AE_STATE["ExposureValue"] = 0.0
        print(f"Init: Expo={AE_STATE['ExposureTime']}us, "
              f"Gain={AE_STATE['AnalogueGain']:.2f}x, "
              f"Locked={AE_STATE['AeLocked']}")
    except Exception:
        print("Warning: could not read initial AE metadata")

    provider = make_frame_provider(cam)

    streamer = MjpegStreamer(
        frame_provider=provider,
        port=5000,
        custom_template=HTML_PAGE,
        custom_routes={
            "/stats":     make_route_stats(),
            "/set_ev":    make_route_set_ev(cam),
            "/enable_ae": make_route_enable_ae(cam),
            "/lock_ae":   make_route_lock_ae(cam),
        },
    )
    streamer.start()
    print("AE debug ready at http://0.0.0.0:5000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down ...")

    streamer.stop()
    cam.release()


if __name__ == "__main__":
    main()
