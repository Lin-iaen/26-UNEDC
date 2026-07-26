#!/usr/bin/env python3
"""相机总控台 —— 参数 / 画幅 / 推流 的一站式手动验证工具

用法：
    source venv/bin/activate
    python tests/test_camera_console.py
    → 浏览器打开 http://<pi-ip>:5000

设计要点：**每一项调整都给出客观证据**，而不是"看起来好像变了"。

  - 曝光 / 增益  → 直接回读 libcamera metadata，设定值 vs 实测值并排显示
  - 亮度/对比度/饱和度 → ISP 控制项没有 metadata 回读，改为实时统计画面的
                          均值 / 标准差 / 饱和度，数值动了才算生效
  - 画幅(FOV)    → 显示 ScalerCrop 与占满传感器的百分比。IMX219 mode 0 是
                    2× 中心裁切，切过去这个数字会掉到约 25%
  - 推流         → 推流 FPS 与采集 FPS 分开统计，前者掉后者不掉 = 编码/网络瓶颈
"""

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

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

# 滑条控制项：key → (最小值, 最大值, 步长, 是否可从 metadata 回读)
CONTROLS = {
    "ExposureTime":  (100, 66000, 100, True),
    "AnalogueGain":  (1.0, 16.0, 0.1, True),
    "Brightness":    (-1.0, 1.0, 0.05, False),
    "Contrast":      (0.0, 8.0, 0.1, False),
    "Saturation":    (0.0, 8.0, 0.1, False),
    "Sharpness":     (0.0, 16.0, 0.1, False),
    "ExposureValue": (-8.0, 8.0, 0.5, False),
}

# 用户设定的值（对照组，用来和硬件实测值比对）
REQUESTED: dict[str, float] = {}

# 输出分辨率候选（主码流尺寸，与传感器模式互不影响）
OUTPUT_SIZES = [(320, 240), (640, 480), (800, 600), (1280, 720), (1640, 1232), (1920, 1080)]

STATE = {
    "vflip": False,
    "hflip": False,
    "mode": None,          # 当前 sensor mode 序号，None = 启动时的默认配置
    "out": None,           # 当前输出分辨率序号，None = 默认 640x480
    "stream_fps": 0.0,
    "frame_stats": {"mean": 0.0, "std": 0.0, "sat": 0.0},
    "shape": "--",
    "last_capture": "",
}

_stream_probe = {"n": 0, "t": time.perf_counter()}
_capture_probe = {"id": 0, "t": time.perf_counter(), "fps": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# 帧提供者
# ══════════════════════════════════════════════════════════════════════════════

def make_frame_provider(cam: Camera):
    counter = {"n": 0, "last_id": -1}

    def provider() -> np.ndarray | None:
        # 相机 21fps，推流循环跑得比它快得多。没有新帧就直接跳过，否则会把同一帧
        # 反复翻转、统计、JPEG 编码 —— 分辨率一高就是这里把 CPU 吃光的。
        fid = cam.frame_id
        if fid == counter["last_id"]:
            return None
        counter["last_id"] = fid

        frame = cam.read()
        if frame is None:
            return None

        if STATE["vflip"] and STATE["hflip"]:
            frame = cv2.flip(frame, -1)
        elif STATE["vflip"]:
            frame = cv2.flip(frame, 0)
        elif STATE["hflip"]:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        STATE["shape"] = f"{w}x{h}"

        # 画面统计每 5 帧算一次即可，避免拖慢推流
        counter["n"] += 1
        if counter["n"] % 5 == 0:
            small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            STATE["frame_stats"] = {
                "mean": round(float(gray.mean()), 1),
                "std": round(float(gray.std()), 1),
                "sat": round(float(hsv[:, :, 1].mean()), 1),
            }

        # 推流帧率
        _stream_probe["n"] += 1
        now = time.perf_counter()
        dt = now - _stream_probe["t"]
        if dt >= 1.0:
            STATE["stream_fps"] = round(_stream_probe["n"] / dt, 1)
            _stream_probe["n"] = 0
            _stream_probe["t"] = now

        cv2.putText(frame, f"{w}x{h}  {STATE['stream_fps']:.0f}fps", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        return frame

    return provider


def capture_fps(cam: Camera) -> float:
    """真实采集帧率：由 frame_id 增量算出，与浏览器是否连着无关。"""
    now = time.perf_counter()
    fid = cam.frame_id
    dt = now - _capture_probe["t"]
    if dt >= 0.5:
        _capture_probe["fps"] = round((fid - _capture_probe["id"]) / dt, 1)
        _capture_probe["id"] = fid
        _capture_probe["t"] = now
    return _capture_probe["fps"]


# ══════════════════════════════════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════════════════════════════════

def make_route_stats(cam: Camera, full_area: int):
    def handler(**kwargs):
        md = cam.get_metadata()
        crop = md.get("ScalerCrop")
        fov = round(100.0 * (crop[2] * crop[3]) / full_area, 1) if crop and full_area else None
        return jsonify({
            "streamFps": STATE["stream_fps"],
            "captureFps": capture_fps(cam),
            "frameId": cam.frame_id,
            "shape": STATE["shape"],
            "mode": STATE["mode"],
            "out": STATE["out"],
            "vflip": STATE["vflip"],
            "hflip": STATE["hflip"],
            "requested": REQUESTED,
            "stats": STATE["frame_stats"],
            "lastCapture": STATE["last_capture"],
            "time": datetime.now().strftime("%H:%M:%S"),
            "measured": {
                "ExposureTime": md.get("ExposureTime"),
                "AnalogueGain": round(md["AnalogueGain"], 2) if "AnalogueGain" in md else None,
                "DigitalGain": round(md["DigitalGain"], 2) if "DigitalGain" in md else None,
                "Lux": round(md["Lux"], 1) if "Lux" in md else None,
                "ColourTemperature": md.get("ColourTemperature"),
                "FrameDuration": md.get("FrameDuration"),
                "AeState": md.get("AeState"),
                "ScalerCrop": list(crop) if crop else None,
                "FovPercent": fov,
            },
        })
    return handler


def make_route_set(cam: Camera):
    def handler(**kwargs):
        applied = {}
        for key in request.args:
            match = next((c for c in CONTROLS if c.lower() == key.lower()), None)
            if match is None:
                continue
            try:
                val = float(request.args[key])
            except ValueError:
                continue
            if match == "ExposureTime":
                val = int(val)
            cam.set_params({match: val})
            REQUESTED[match] = val
            applied[match] = val
        # 手动设曝光/增益意味着退出自动曝光
        if "ExposureTime" in applied or "AnalogueGain" in applied:
            cam.set_params({"AeEnable": False})
        return jsonify({"ok": True, "applied": applied})
    return handler


def make_route_ae_on(cam: Camera):
    def handler(**kwargs):
        cam.set_params({"AeEnable": True})
        REQUESTED.pop("ExposureTime", None)
        REQUESTED.pop("AnalogueGain", None)
        return jsonify({"ok": True})
    return handler


def make_route_ae_lock(cam: Camera):
    def handler(**kwargs):
        """锁死当前 AE 收敛结果：先回读实测值，再关掉 AE 并写回去。"""
        md = cam.get_metadata()
        exp = md.get("ExposureTime", 20000)
        gain = md.get("AnalogueGain", 1.0)
        cam.set_params({"AeEnable": False, "ExposureTime": exp, "AnalogueGain": gain})
        REQUESTED["ExposureTime"] = exp
        REQUESTED["AnalogueGain"] = round(gain, 2)
        return jsonify({"ok": True, "ExposureTime": exp, "AnalogueGain": gain})
    return handler


def make_route_modes(cam: Camera):
    def handler(**kwargs):
        return jsonify([
            {"i": i, "size": list(m["size"]), "bit_depth": m["bit_depth"],
             "fps": round(m["fps"], 0)}
            for i, m in enumerate(cam.sensor_modes)
        ])
    return handler


def make_route_mode(cam: Camera):
    def handler(**kwargs):
        mode_id = int(kwargs.get("mode_id", 0))
        cam.switch_sensor_mode(mode_id)
        STATE["mode"] = mode_id
        return jsonify({"ok": True, "mode": mode_id})
    return handler


def make_route_sizes():
    def handler(**kwargs):
        return jsonify([{"i": i, "w": w, "h": h} for i, (w, h) in enumerate(OUTPUT_SIZES)])
    return handler


def make_route_size(cam: Camera):
    def handler(**kwargs):
        idx = int(kwargs.get("size_id", 0))
        if not 0 <= idx < len(OUTPUT_SIZES):
            return jsonify({"ok": False, "error": "bad index"}), 400
        cam.set_output_size(OUTPUT_SIZES[idx])
        STATE["out"] = idx
        return jsonify({"ok": True, "size": list(OUTPUT_SIZES[idx])})
    return handler


def make_route_flip():
    def handler(**kwargs):
        axis = kwargs.get("axis", "v")
        if axis in ("v", "h"):
            STATE[f"{axis}flip"] = not STATE[f"{axis}flip"]
        return jsonify({"ok": True, "vflip": STATE["vflip"], "hflip": STATE["hflip"]})
    return handler


def make_route_capture(cam: Camera):
    def handler(**kwargs):
        frame = cam.read()
        if frame is None:
            return jsonify({"ok": False, "error": "no frame"}), 503
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        path = SAMPLES_DIR / f"console_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        STATE["last_capture"] = path.name
        return jsonify({"ok": True, "path": path.name})
    return handler


# ══════════════════════════════════════════════════════════════════════════════
# 页面
# ══════════════════════════════════════════════════════════════════════════════

def build_page() -> str:
    sliders = []
    for key, (lo, hi, step, readback) in CONTROLS.items():
        tag = '<span class="rb">可回读</span>' if readback else ""
        sliders.append(f"""
<label>{key} {tag}<span class="set" id="set_{key}">--</span></label>
<input type="range" id="sl_{key}" min="{lo}" max="{hi}" step="{step}"
       oninput="setVal('{key}', this.value)">""")

    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>相机总控台</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:400px;overflow-y:auto;padding:12px;background:#1a1a1a;border-right:1px solid #333}
#main{flex:1;display:flex;align-items:center;justify-content:center;background:#000;padding:4px}
img{max-width:100%;max-height:100vh;object-fit:contain;display:block}
h3{margin:14px 0 6px;font-size:12px;color:#6cf;border-bottom:1px solid #333;padding-bottom:3px}
.row{display:flex;justify-content:space-between;font-size:12px;margin:3px 0}
.row .k{color:#888}.row .v{color:#0f0}
.row .v.warn{color:#fa0}
label{display:block;margin:10px 0 2px;font-size:11px;color:#aaa}
input[type=range]{width:100%}
.set{float:right;color:#0f0;font-size:11px}
.rb{font-size:9px;color:#6cf;border:1px solid #245;padding:0 3px;margin-left:4px}
.btn{display:inline-block;margin:3px 2px;padding:5px 9px;background:#2a2a2a;color:#ddd;
     border:1px solid #444;cursor:pointer;font-size:11px;font-family:monospace}
.btn:hover{background:#3a3a3a}
.btn.on{background:#1a3a1a;border-color:#2a6a2a;color:#8f8}
.btn.cur{background:#1a2a4a;border-color:#3a5a8a;color:#8cf}
#msg{font-size:11px;color:#fa0;min-height:14px;margin-top:6px}
.hint{font-size:10px;color:#777;line-height:1.5;margin:2px 0 6px}
.hint b{color:#fa0}
</style></head><body>
<div id="panel">

<h3>推流</h3>
<div class="row"><span class="k">推流 FPS（编码后)</span><span class="v" id="s_sfps">--</span></div>
<div class="row"><span class="k">采集 FPS（驱动层)</span><span class="v" id="s_cfps">--</span></div>
<div class="row"><span class="k">帧序号</span><span class="v" id="s_fid">--</span></div>
<div class="row"><span class="k">输出分辨率</span><span class="v" id="s_shape">--</span></div>

<h3>硬件实测 (libcamera metadata)</h3>
<div class="row"><span class="k">ExposureTime</span><span class="v" id="m_exp">--</span></div>
<div class="row"><span class="k">AnalogueGain</span><span class="v" id="m_gain">--</span></div>
<div class="row"><span class="k">DigitalGain</span><span class="v" id="m_dgain">--</span></div>
<div class="row"><span class="k">Lux</span><span class="v" id="m_lux">--</span></div>
<div class="row"><span class="k">色温</span><span class="v" id="m_ct">--</span></div>
<div class="row"><span class="k">帧间隔</span><span class="v" id="m_fd">--</span></div>
<div class="row"><span class="k">AeState</span><span class="v" id="m_ae">--</span></div>

<h3>输出分辨率（改 read() 的画面尺寸)</h3>
<div class="hint">主码流尺寸 —— <b>只有改这个左上角绿字才会变</b>。分辨率越高 JPEG
编码越吃 CPU,推流 FPS 会掉。视场基本不变,但宽高比不同于 4:3 时会做裁切
(如 16:9 会切掉上下)。</div>
<div id="size_buttons"></div>

<h3>传感器模式（改视场 FOV)</h3>
<div class="hint">选传感器读出哪块区域 + 最高帧率。ISP 会缩放到上面的输出分辨率,
<b>所以切模式不改变画面尺寸</b>,只改视场和采集帧率 —— 看 ScalerCrop 判断。
默认已锁全画幅。</div>
<div class="row"><span class="k">ScalerCrop</span><span class="v" id="m_crop">--</span></div>
<div class="row"><span class="k">占传感器面积</span><span class="v" id="m_fov">--</span></div>
<div class="row"><span class="k">当前 sensor mode</span><span class="v" id="s_mode">--</span></div>
<div id="mode_buttons"></div>

<h3>画面统计（验证 ISP 参数)</h3>
<div class="row"><span class="k">亮度均值 ← Brightness</span><span class="v" id="st_mean">--</span></div>
<div class="row"><span class="k">标准差 ← Contrast</span><span class="v" id="st_std">--</span></div>
<div class="row"><span class="k">饱和度均值 ← Saturation</span><span class="v" id="st_sat">--</span></div>

<h3>参数</h3>
__SLIDERS__

<h3>自动曝光</h3>
<span class="btn" onclick="hit('/ae/on')">开启 AE</span>
<span class="btn" onclick="hit('/ae/lock')">锁死当前值</span>

<h3>翻转 / 拍照</h3>
<span class="btn" id="b_vflip" onclick="hit('/flip/v')">↕ V-Flip</span>
<span class="btn" id="b_hflip" onclick="hit('/flip/h')">↔ H-Flip</span>
<span class="btn" onclick="hit('/capture')">📷 拍照存盘</span>
<div class="row"><span class="k">最近存盘</span><span class="v" id="s_cap">--</span></div>
<div id="msg"></div>
</div>

<div id="main"><img src="/video_feed"></div>

<script>
var KEYS = __KEYS__;
var dragging = null;

function msg(t){ document.getElementById('msg').textContent = t; }

// 滑条节流：oninput 每移动一像素就触发一次，不加限制会把浏览器的连接槽打满
// （MJPEG 长连接本身已占掉一个），/stats 排队堆积后整个页面发僵。
// 合并中间值，保证最后一个值一定发出去。
var pending = {}, timer = {};
function setVal(k, v){
  dragging = k;
  document.getElementById('set_'+k).textContent = v;
  pending[k] = v;
  if(timer[k]) return;
  timer[k] = setTimeout(function(){
    timer[k] = null;
    var val = pending[k];
    fetch('/set?'+k+'='+val).catch(e=>msg('设置失败: '+e));
    setTimeout(function(){ if(dragging===k) dragging=null; }, 1200);
  }, 80);
}

function hit(url){
  fetch(url).then(r=>r.json()).then(function(d){
    msg(d.ok ? url+' OK' : url+' 失败: '+(d.error||''));
    poll();
  }).catch(e=>msg(url+' 请求失败: '+e));
}

function fmt(v, unit){ return (v===null||v===undefined) ? '--' : v+(unit||''); }

function poll(){
  fetch('/stats').then(r=>r.json()).then(function(d){
    document.getElementById('s_sfps').textContent = d.streamFps;
    document.getElementById('s_cfps').textContent = d.captureFps;
    document.getElementById('s_fid').textContent = d.frameId;
    document.getElementById('s_shape').textContent = d.shape;
    document.getElementById('s_mode').textContent = (d.mode===null?'默认配置':d.mode);
    document.getElementById('s_cap').textContent = d.lastCapture || '--';

    var m = d.measured;
    document.getElementById('m_exp').textContent  = fmt(m.ExposureTime,' us');
    document.getElementById('m_gain').textContent = fmt(m.AnalogueGain,' x');
    document.getElementById('m_dgain').textContent= fmt(m.DigitalGain,' x');
    document.getElementById('m_lux').textContent  = fmt(m.Lux);
    document.getElementById('m_ct').textContent   = fmt(m.ColourTemperature,' K');
    document.getElementById('m_fd').textContent   = fmt(m.FrameDuration,' us');
    document.getElementById('m_ae').textContent   = fmt(m.AeState);
    document.getElementById('m_crop').textContent = m.ScalerCrop ? m.ScalerCrop.join(', ') : '--';
    document.getElementById('m_fov').textContent  = fmt(m.FovPercent,' %');

    document.getElementById('st_mean').textContent = d.stats.mean;
    document.getElementById('st_std').textContent  = d.stats.std;
    document.getElementById('st_sat').textContent  = d.stats.sat;

    document.getElementById('b_vflip').className = 'btn' + (d.vflip?' on':'');
    document.getElementById('b_hflip').className = 'btn' + (d.hflip?' on':'');

    // 回填滑条，但不要抢正在拖动的那一根
    KEYS.forEach(function(k){
      if(k === dragging) return;
      var v = d.requested[k];
      if(v === undefined && m[k] !== undefined && m[k] !== null) v = m[k];
      if(v !== undefined && v !== null){
        document.getElementById('set_'+k).textContent = v;
        document.getElementById('sl_'+k).value = v;
      }
    });

    document.querySelectorAll('#mode_buttons .btn').forEach(function(b){
      b.className = 'btn' + (parseInt(b.dataset.i)===d.mode ? ' cur' : '');
    });
    document.querySelectorAll('#size_buttons .btn').forEach(function(b){
      b.className = 'btn' + (parseInt(b.dataset.i)===d.out ? ' cur' : '');
    });
  }).catch(e=>msg('/stats 请求失败: '+e));
}

fetch('/modes').then(r=>r.json()).then(function(modes){
  document.getElementById('mode_buttons').innerHTML = modes.map(function(m){
    return '<span class="btn" data-i="'+m.i+'" onclick="hit(\\'/mode/'+m.i+'\\')">'
         + m.i+': '+m.size[0]+'x'+m.size[1]+' '+m.bit_depth+'bit @'+m.fps+'</span>';
  }).join('');
  poll();
});

fetch('/sizes').then(r=>r.json()).then(function(sizes){
  document.getElementById('size_buttons').innerHTML = sizes.map(function(s){
    return '<span class="btn" data-i="'+s.i+'" onclick="hit(\\'/size/'+s.i+'\\')">'
         + s.w+'x'+s.h+'</span>';
  }).join('');
});

poll();
setInterval(poll, 1000);
</script>
</body></html>""".replace("__SLIDERS__", "".join(sliders)) \
                 .replace("__KEYS__", str(list(CONTROLS.keys())).replace("'", '"'))


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cam = Camera(vflip=False, hflip=False)
    cam.start()
    time.sleep(1.5)   # 等 AE 收敛

    full = max((m["size"][0] * m["size"][1] for m in cam.sensor_modes), default=0)

    md = cam.get_metadata()
    if md:
        print(f"启动实测: Expo={md.get('ExposureTime')}us "
              f"Gain={md.get('AnalogueGain', 0):.2f}x "
              f"Crop={md.get('ScalerCrop')}")

    streamer = MjpegStreamer(
        frame_provider=make_frame_provider(cam),
        port=5000,
        max_fps=60.0,          # 高于采集率，实际节奏由 provider 的去重决定
        custom_template=build_page(),
        custom_routes={
            "/stats":              make_route_stats(cam, full),
            "/set":                make_route_set(cam),
            "/ae/on":              make_route_ae_on(cam),
            "/ae/lock":            make_route_ae_lock(cam),
            "/modes":              make_route_modes(cam),
            "/mode/<int:mode_id>": make_route_mode(cam),
            "/sizes":              make_route_sizes(),
            "/size/<int:size_id>": make_route_size(cam),
            "/flip/<axis>":        make_route_flip(),
            "/capture":            make_route_capture(cam),
        },
    )
    streamer.start()
    print("相机总控台就绪 → http://<pi-ip>:5000   (Ctrl-C 退出)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在退出 ...")
    finally:
        streamer.stop()
        cam.release()


if __name__ == "__main__":
    main()
