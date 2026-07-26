#!/usr/bin/env python3
"""
矩形检测测试脚本 —— 双窗口调试版
===================================

检测由 1.8cm 黑色电工胶带在 A4 白纸上拼接的矩形，识别内外两层轮廓。
提供网页参数滑块、双窗口（检测/二值化）实时调试视图。

用法：
    python tests/test_rectangle_detect.py
    → 打开 http://<pi-ip>:5000，拖动滑块实时观察过滤效果
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.drivers import Camera
from src.vision import MjpegStreamer

# ══════════════════════════════════════════════════════════════════════════════
# 全局可调参数 —— 启动时用 AE 实测值填充曝光/增益，其余给初值
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = {
    # 摄像机参数 —— start() 后用 capture_metadata() 覆盖实际值
    "ExposureTime": 20000,
    "AnalogueGain": 1.0,
    # 检测过滤参数
    "MinArea":        500,    # 最小轮廓面积 (px²)
    "MaxAspectRatio": 3.0,    # 最大长宽比（超过视为长条，丢弃）
    "MinContrast":    30,     # 最小对比度差值（白纸灰度 - 胶带灰度）
    "GlobalThresh":   120,    # 全局二值化阈值（AE 锁定后光照恒定）
}

# ══════════════════════════════════════════════════════════════════════════════
# 绘图颜色 (BGR)
# ══════════════════════════════════════════════════════════════════════════════

CLR_OUTER   = (0, 255, 0)     # 绿色 —— 外层轮廓（胶带外沿）
CLR_INNER   = (255, 0, 0)     # 蓝色 —— 内层轮廓（胶带内沿）
CLR_DIAG    = (0, 0, 255)     # 红色 —— 对角线
CLR_LABEL   = (255, 255, 0)   # 青色 —— 通过检测的标注
CLR_REJECT  = (80, 80, 80)    # 深灰 —— 被过滤掉的轮廓
CLR_TEXT    = (200, 200, 200) # 浅灰 —— 过滤原因文字
CLR_STATUS  = (0, 255, 255)   # 黄色 —— 状态栏

# ══════════════════════════════════════════════════════════════════════════════
# Web UI
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Rectangle Detect</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:320px;overflow-y:auto;padding:10px;background:#1a1a1a;border-right:1px solid #333}
#main{flex:1;display:flex;align-items:center;justify-content:center;padding:4px}
img{max-width:100%;max-height:100%;object-fit:contain}
label{display:block;margin:6px 0 2px;font-size:11px;color:#aaa}
input[type=range]{width:100%}
.val{font-size:10px;color:#0f0;float:right}
.btn{display:inline-block;margin:2px;padding:3px 8px;background:#333;color:#fff;
     border:1px solid #555;cursor:pointer;font-size:10px;text-decoration:none}
.btn:hover{background:#555}
h3{margin:10px 0 6px;font-size:12px;border-bottom:1px solid #333;padding-bottom:3px}
.stats{font-size:10px;color:#666;margin-top:6px}
.section{margin-bottom:4px}
</style></head><body>
<div id="panel">

<h3>📷 Camera</h3>
<label>曝光时间 <span class="val" id="v_ExposureTime">--</span> µs</label>
<input type="range" id="sl_ExposureTime" min="39" max="66666"
       oninput="setParam('ExposureTime',parseInt(this.value))">

<label>模拟增益 <span class="val" id="v_AnalogueGain">--</span> ×</label>
<input type="range" id="sl_AnalogueGain" min="1" max="16" step="0.1"
       oninput="setParam('AnalogueGain',parseFloat(this.value))">
<a class="btn" href="/reset_ae">↻ 恢复自动曝光</a>

<h3>🔍 检测过滤</h3>
<label>最小面积 <span class="val" id="v_MinArea">--</span> px²</label>
<input type="range" id="sl_MinArea" min="100" max="10000"
       oninput="setParam('MinArea',parseInt(this.value))">

<label>最大长宽比 <span class="val" id="v_MaxAspectRatio">--</span></label>
<input type="range" id="sl_MaxAspectRatio" min="1" max="10" step="0.1"
       oninput="setParam('MaxAspectRatio',parseFloat(this.value))">

<label>最小对比度 <span class="val" id="v_MinContrast">--</span></label>
<input type="range" id="sl_MinContrast" min="10" max="100"
       oninput="setParam('MinContrast',parseInt(this.value))">

<h3>🧠 二值化</h3>
<label>阈值 <span class="val" id="v_GlobalThresh">--</span></label>
<input type="range" id="sl_GlobalThresh" min="0" max="255"
       oninput="setParam('GlobalThresh',parseInt(this.value))">

<div class="stats">
帧率: <span id="fps">--</span> FPS<br>
检测数: <span id="rects">--</span><br>
<span id="ts">--</span>
</div>
</div>

<div id="main">
<img src="/video_feed" id="stream">
</div>

<script>
const ALL_KEYS = ["ExposureTime","AnalogueGain","MinArea","MaxAspectRatio","MinContrast","GlobalThresh"];
// 将 key 的首字母转小写（Python dict key 的格式），因为 /stats 返回的 JSON key 是首字母小写
function toJsonKey(k){return k.charAt(0).toLowerCase()+k.slice(1);}

function setParam(name, val){
  document.getElementById('v_'+name).textContent = val;
  document.getElementById('sl_'+name).value = val;
  fetch('/set?'+toJsonKey(name)+'='+val);
}

function poll(){
  fetch('/stats').then(r=>r.json()).then(function(d){
    ALL_KEYS.forEach(function(k){
      var jk = toJsonKey(k);
      var v = d[jk];
      if(v !== undefined && v !== null){
        document.getElementById('v_'+k).textContent = v;
        document.getElementById('sl_'+k).value = v;
      }
    });
    document.getElementById('fps').textContent = d.fps;
    document.getElementById('ts').textContent = d.time;
    document.getElementById('rects').textContent = d.rects;
  });
}
poll();
setInterval(poll, 2000);
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 检测 pipeline
# ══════════════════════════════════════════════════════════════════════════════


def _is_rectangle(approx: np.ndarray) -> bool:
    """approxPolyDP 结果是否为凸四边形"""
    return len(approx) == 4 and cv2.isContourConvex(approx)


def _aspect_ratio(cnt: np.ndarray) -> float:
    """轮廓的宽高比，始终 ≥ 1（宽矮或瘦高都归一化到同一尺度）"""
    x, y, w, h = cv2.boundingRect(cnt)
    if min(w, h) == 0:
        return 999.0
    return max(w / h, h / w)


def _contrast_check(
    gray: np.ndarray,
    outer_approx: np.ndarray,
    inner_approx: np.ndarray,
) -> float:
    """
    返回白纸区域与胶带区域的灰度差值（>0 表示白纸更亮）。

    性能：先求外轮廓 bounding rect，仅截取这块极小的 ROI 灰度图，
    然后在该 ROI 上创建 mask 并做均值计算，避免全图级 640×480 内存遍历。
    对于 200×200 的矩形框，ROI 面积仅为全图的 ~1/7。
    """
    # 外轮廓的外接矩形 → ROI 坐标
    x, y, w, h = cv2.boundingRect(outer_approx)
    # 像素偏移量 —— 用于将原图坐标平移到 ROI 内部
    ox, oy = x, y

    # 截取 ROI 灰度块（仅此小块，不是全图）
    roi_gray = gray[y:y+h, x:x+w]

    # 将轮廓坐标平移到 ROI 局部坐标系
    outer_local = outer_approx.copy()
    inner_local = inner_approx.copy()
    outer_local[:, 0, 0] -= ox  # x
    outer_local[:, 0, 1] -= oy  # y
    inner_local[:, 0, 0] -= ox
    inner_local[:, 0, 1] -= oy

    # 在极小 ROI 上创建 mask —— 内存从 640×480 降到 w×h
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(inner_mask, [inner_local], -1, 255, thickness=cv2.FILLED)

    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(outer_mask, [outer_local], -1, 255, thickness=cv2.FILLED)
    tape_mask = cv2.subtract(outer_mask, inner_mask)

    inner_gray = cv2.mean(roi_gray, mask=inner_mask)[0]
    outer_gray = cv2.mean(roi_gray, mask=tape_mask)[0]

    return inner_gray - outer_gray


def _order_corners(approx: np.ndarray) -> np.ndarray:
    """
    将 approxPolyDP 返回的四个顶点排序为：
    [左上, 右上, 右下, 左下]（顺时针）。
    approxPolyDP 不保证点顺序一致性，所以需要排序。
    """
    pts = approx.reshape(4, 2)
    # 先按 y 再按 x 排序 → 上半顶点在前、下半在后
    idx = np.lexsort((pts[:, 0], pts[:, 1]))
    pts = pts[idx]
    # 上半两个顶点按 x 排序 → 左上、右上
    top = pts[:2][np.argsort(pts[:2, 0])]
    # 下半两个顶点按 x 降序 → 右下、左下
    bottom = pts[2:][np.argsort(pts[2:, 0])[::-1]]
    return np.vstack([top, bottom])


def process_frame(frame: np.ndarray, params: dict) -> tuple[np.ndarray, int]:
    """
    主检测函数：在一帧上做矩形检测 + 双窗口合成。
    返回合成后的 BGR 帧（检测结果 | 调试视图）。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 解读参数 ──────────────────────────────────────────────────────
    min_area   = int(params.get("MinArea", 500))
    max_ar     = float(params.get("MaxAspectRatio", 3.0))
    min_contrast = int(params.get("MinContrast", 30))
    global_thresh = int(params.get("GlobalThresh", 120))

    # ── 全局二值化 ──────────────────────────────────────────────────
    # AE 锁定后光照归一化，固定阈值稳定可靠。
    # 产生实心二值区域（非 Canny 的 1px 骨架），确保 findContours
    # 能形成闭合轮廓，识别率大幅提高。
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, global_thresh, 255, cv2.THRESH_BINARY_INV,
    )

    debug_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # detection_view 从原始帧拷贝开始，不加任何预绘——只有检测结果才画
    detection_view = frame.copy()

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )

    # 调试视图：只画所有原始轮廓（灰色细线），不加任何文字
    cv2.drawContours(debug_view, contours, -1, (70, 70, 70), 1)

    if hierarchy is None:
        return _compose_views(detection_view, debug_view, 0), 0

    hierarchy = hierarchy[0]
    found_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    # 只记录"通过四边形检测但被后续过滤淘汰"的候选——减少无关噪点标注
    rejected: list[tuple[np.ndarray, str]] = []

    for idx, (cnt, hinfo) in enumerate(zip(contours, hierarchy)):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 不是四边形 → 太常见（墙上纹理、纸边角），不加标注
        if not _is_rectangle(approx):
            continue

        ar = _aspect_ratio(approx)
        # 长宽比过大 → 可能是纸边、桌沿，标注原因
        if ar > max_ar:
            rejected.append((approx, f"AR={ar:.1f}>{max_ar}"))
            continue

        # 找子轮廓（胶带内沿）
        child_idx = hinfo[2]
        if child_idx == -1:
            # 无子轮廓 → 可能是纸角而不是胶带框，不加标注
            continue

        child_cnt = contours[child_idx]
        child_epsilon = 0.02 * cv2.arcLength(child_cnt, True)
        child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)

        if not _is_rectangle(child_approx):
            continue

        # 对比度验证：用原始灰度图计算"白纸 vs 胶带"的灰度差
        # 低曝光下白纸灰度可能只有 80，但胶带更低，差值仍然显著
        contrast = _contrast_check(gray, approx, child_approx)
        if contrast <= min_contrast:
            rejected.append((approx, f"C={contrast:.0f}<{min_contrast}"))
            continue

        found_pairs.append((approx, child_approx))

    # ── 绘制检测视图：只画最终结果和相关过滤信息 ─────────────────────

    # 被过滤的候选（灰线 + 原因标注，放在轮廓外侧避免重叠）
    for cnt, reason in rejected:
        cv2.drawContours(detection_view, [cnt], -1, CLR_REJECT, 2)
        # 文字放在轮廓右下角外侧，避开与其他轮廓的文字碰撞
        x, y, cw, ch = cv2.boundingRect(cnt)
        cv2.putText(detection_view, reason, (x + cw + 4, y + ch // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_TEXT, 1, cv2.LINE_AA)

    # 通过的矩形
    for outer, inner in found_pairs:
        cv2.drawContours(detection_view, [outer], -1, CLR_OUTER, 2)
        cv2.drawContours(detection_view, [inner], -1, CLR_INNER, 2)
        ordered = _order_corners(inner)
        cv2.line(detection_view, tuple(ordered[0]), tuple(ordered[2]),
                 CLR_DIAG, 2)
        cv2.line(detection_view, tuple(ordered[1]), tuple(ordered[3]),
                 CLR_DIAG, 2)

    return _compose_views(detection_view, debug_view, len(found_pairs)), len(found_pairs)


def _compose_views(
    detection: np.ndarray,
    debug: np.ndarray,
    rect_count: int,
) -> np.ndarray:
    """并排合成 + 左上角状态标记。"""
    combined = np.hstack([detection, debug])

    # 仅左上角一行状态
    colour = (0, 255, 0) if rect_count else (0, 0, 255)
    status = f"R:{rect_count}" if rect_count else "--"
    cv2.putText(combined, status, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

_frame_count = 0
_fps = 0.0
_last_ts = time.perf_counter()
_rect_count = 0
_ae_active = False   # True 表示 AE 正在自动调节，需从 metadata 同步参数


def main() -> None:
    global _frame_count, _fps, _last_ts, _rect_count

    cam = Camera(vflip=True)
    cam.start()

    # ── 读取自动曝光稳定后的实际值，然后锁定 ─────────────────────────
    # 不等 AE 稳定就直接用默认值的话，锁定的参数和实际不匹配，
    # 画面会突然跳变。所以先在后台采集几秒让 AE 收敛。
    time.sleep(2.0)
    # 通过 picamera2 底层的 capture_metadata 获取当前实际值
    try:
        from picamera2 import Picamera2
        metadata = cam._cam.capture_metadata()
        actual_exp = metadata.get("ExposureTime", 10000)
        actual_gain = metadata.get("AnalogueGain", 4.5)
        PARAMS["ExposureTime"] = actual_exp
        PARAMS["AnalogueGain"] = actual_gain
        # 锁定曝光/增益，防止 AE 后续改动导致二值化阈值漂移
        cam.set_params({
            "ExposureTime": actual_exp,
            "AnalogueGain": actual_gain,
        })
        print(f"AE locked: Exposure={actual_exp}us, Gain={actual_gain:.2f}x")
    except Exception:
        print("Warning: could not read AE metadata, using defaults")

    def frame_provider() -> np.ndarray | None:
        global _frame_count, _fps, _last_ts, _rect_count, _ae_active
        frame = cam.read()
        if frame is None:
            return None

        # AE 激活时从 metadata 实时同步曝光/增益到 UI 参数
        if _ae_active:
            try:
                metadata = cam._cam.capture_metadata()
                PARAMS["ExposureTime"] = metadata.get("ExposureTime", 10000)
                PARAMS["AnalogueGain"] = metadata.get("AnalogueGain", 4.5)
            except Exception:
                pass

        try:
            result, _rect_count = process_frame(frame, PARAMS)
        except Exception as e:
            # process_frame 异常 → 回退到原始帧，保证 MJPEG 不断流
            import traceback
            traceback.print_exc()
            result = np.hstack([frame, np.zeros_like(frame)])
            _rect_count = -1

        _frame_count += 1
        now = time.perf_counter()
        if now - _last_ts >= 1.0:
            _fps = _frame_count / (now - _last_ts)
            _frame_count = 0
            _last_ts = now

        return result

    streamer = MjpegStreamer(
        frame_provider=frame_provider,
        port=5000,
        custom_template=HTML_PAGE,
        custom_routes={
            "/stats": _make_route_stats(),
            "/set": _make_route_set(cam),
            "/reset_ae": _make_route_reset_ae(cam),
        },
    )
    streamer.start()
    print("Rectangle detector ready at http://0.0.0.0:5000")
    print("  Left panel  = parameter sliders")
    print("  Left half   = detection result + reject labels")
    print("  Right half  = binary threshold debug view")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down ...")

    streamer.stop()
    cam.release()


# ══════════════════════════════════════════════════════════════════════════════
# 路由工厂
# ══════════════════════════════════════════════════════════════════════════════

def _make_route_stats():
    """返回 /stats JSON：当前参数值 + 帧率"""
    def handler(**kwargs):
        import datetime
        return {
            "fps": round(_fps, 1),
            "rects": _rect_count,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            **{k[0].lower() + k[1:]: v for k, v in PARAMS.items()},
        }
    return handler


def _make_route_set(cam: Camera):
    """处理 /set?key=value，更新 PARAMS，相机参数同步到硬件"""
    from flask import jsonify, request
    def handler(**kwargs):
        global _ae_active
        for key in request.args:
            raw = request.args[key]
            # 尝试转数字
            try:
                val = float(raw)
                val = int(val) if val == int(val) and "." not in raw else val
            except ValueError:
                continue
            # 找到 PARAMS 中匹配的 key（大小写不敏感）
            matched = None
            for pk in PARAMS:
                if pk.lower() == key.lower():
                    matched = pk
                    break
            if matched:
                PARAMS[matched] = val
                # 相机参数同步到硬件
                if matched in ("ExposureTime", "AnalogueGain"):
                    cam.set_params({matched: val})
                    _ae_active = False  # 手动调参 → 退出 AE 模式
        return jsonify({"ok": True})
    return handler


def _make_route_reset_ae(cam: Camera):
    """恢复自动曝光：解除 ExposureTime 和 AnalogueGain 手动锁定"""
    from flask import jsonify
    def handler(**kwargs):
        global _ae_active
        # 让 ISP 重新接管曝光和增益
        cam.set_params({"AeEnable": True})
        _ae_active = True
        # 立即回读实际值同步到 UI 参数
        try:
            metadata = cam._cam.capture_metadata()
            PARAMS["ExposureTime"] = metadata.get("ExposureTime", 10000)
            PARAMS["AnalogueGain"] = metadata.get("AnalogueGain", 4.5)
        except Exception:
            pass
        print("AE re-enabled")
        return jsonify({"ok": True})
    return handler


if __name__ == "__main__":
    main()
