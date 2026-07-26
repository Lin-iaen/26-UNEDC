#!/usr/bin/env python3
"""无人值守自检 —— 驱动层与推流层的契约检查，不需要浏览器

用法：
    source venv/bin/activate
    python tests/test_smoke.py

约 30 秒跑完，逐项 PASS/FAIL，全通过退出码为 0。
开网页手动调参之前先跑这个：它能把"相机/推流本身坏了"和"参数调了没效果"
这两类问题区分开。
"""

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.drivers import Camera
from src.vision import MjpegStreamer

PORT = 5051
RESULTS: list[tuple[str, bool, str]] = []

# 绕开 HTTP_PROXY/HTTPS_PROXY：本机代理（clash 之类）会把 127.0.0.1 的请求也
# 接管走，然后回 502，看起来就像推流坏了。
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_get(url: str, timeout: float = 3.0, data: bytes | None = None,
             method: str | None = None):
    req = urllib.request.Request(url, data=data, method=method)
    return _opener.open(req, timeout=timeout)


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s}  {detail}")
    return ok


def wait_frame(cam: Camera, timeout: float = 5.0) -> np.ndarray | None:
    """轮询等首帧 —— _capture_loop 开头有 1 秒 AE 稳定延时，不能只 sleep 一下。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = cam.read()
        if frame is not None:
            return frame
        time.sleep(0.05)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 驱动层
# ══════════════════════════════════════════════════════════════════════════════

def check_camera(cam: Camera) -> None:
    print("\n[1] 相机驱动")

    frame = wait_frame(cam)
    if not check("首帧到达", frame is not None, "5s 内取到帧" if frame is not None else "超时"):
        return

    check("帧格式为 BGR uint8 三通道",
          frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8,
          f"shape={frame.shape} dtype={frame.dtype}")

    # 采集帧率：靠 frame_id 增量，read() 会重复返回同一帧所以不能直接数
    id0, t0 = cam.frame_id, time.perf_counter()
    time.sleep(2.0)
    fps = (cam.frame_id - id0) / (time.perf_counter() - t0)
    check("frame_id 递增且帧率合理", 5.0 < fps < 200.0, f"{fps:.1f} FPS")

    # 画面不是纯黑/纯白死图
    check("画面有内容（非恒定值)", float(frame.std()) > 1.0, f"std={frame.std():.1f}")


def check_params(cam: Camera) -> None:
    print("\n[2] 参数下发是否真的进硬件")

    cam.set_params({"AeEnable": False, "ExposureTime": 8000, "AnalogueGain": 2.0})
    time.sleep(1.5)
    md = cam.get_metadata()
    exp, gain = md.get("ExposureTime"), md.get("AnalogueGain")
    check("ExposureTime 回读匹配 (设 8000)",
          exp is not None and abs(exp - 8000) < 500, f"实测 {exp} us")
    check("AnalogueGain 回读匹配 (设 2.0)",
          gain is not None and abs(gain - 2.0) < 0.3, f"实测 {gain:.2f} x" if gain else "无回读")

    cam.set_params({"ExposureTime": 30000})
    time.sleep(1.5)
    exp2 = cam.get_metadata().get("ExposureTime")
    check("改成 30000 后回读跟随",
          exp2 is not None and abs(exp2 - 30000) < 2000, f"实测 {exp2} us")

    # ISP 控制项没有 metadata 回读，改用画面亮度间接验证
    cam.set_params({"Brightness": -0.6})
    time.sleep(1.2)
    dark = cam.read()
    cam.set_params({"Brightness": 0.6})
    time.sleep(1.2)
    bright = cam.read()
    if dark is not None and bright is not None:
        d, b = float(dark.mean()), float(bright.mean())
        check("Brightness 改变画面亮度", b > d + 5, f"暗={d:.1f} → 亮={b:.1f}")
    else:
        check("Brightness 改变画面亮度", False, "取帧失败")
    cam.set_params({"Brightness": 0.0, "AeEnable": True})


def check_sensor_mode(cam: Camera) -> None:
    print("\n[3] 画幅 / 传感器模式切换")

    modes = cam.sensor_modes
    if not check("sensor_modes 可枚举", len(modes) > 0, f"{len(modes)} 个模式"):
        return

    full = max(m["size"][0] * m["size"][1] for m in modes)

    def fov_percent() -> float | None:
        crop = cam.get_metadata().get("ScalerCrop")
        return 100.0 * crop[2] * crop[3] / full if crop else None

    # 找一个全画幅模式和一个裁切模式来对比
    wide = next((i for i, m in enumerate(modes)
                 if m.get("crop_limits", (0, 0, 0, 0))[2] * m.get("crop_limits", (0, 0, 0, 0))[3]
                 >= full * 0.9), None)
    narrow = next((i for i, m in enumerate(modes)
                   if m.get("crop_limits", (0, 0, 0, 0))[2] * m.get("crop_limits", (0, 0, 0, 0))[3]
                   < full * 0.5), None)

    if wide is None or narrow is None:
        check("找到可对比的宽/窄视场模式", False, "该传感器无裁切模式，跳过 FOV 对比")
        return

    cam.switch_sensor_mode(wide)
    time.sleep(2.0)
    fov_wide = fov_percent()
    frame_wide = cam.read()
    check(f"切到模式 {wide} (全画幅) 后仍出图",
          frame_wide is not None, f"FOV={fov_wide:.0f}%" if fov_wide else "无 crop 信息")

    cam.switch_sensor_mode(narrow)
    time.sleep(2.0)
    fov_narrow = fov_percent()
    frame_narrow = cam.read()
    check(f"切到模式 {narrow} (裁切) 后仍出图",
          frame_narrow is not None, f"FOV={fov_narrow:.0f}%" if fov_narrow else "无 crop 信息")

    if fov_wide and fov_narrow:
        check("ScalerCrop 证明视场确实变窄",
              fov_narrow < fov_wide * 0.7, f"{fov_wide:.0f}% → {fov_narrow:.0f}%")

    check("非法模式号不崩溃", _no_raise(cam.switch_sensor_mode, 999), "已忽略并告警")

    # 切模式只改视场，不该改输出尺寸
    before = cam.output_size
    cam.switch_sensor_mode(wide)
    wait_frame(cam)
    check("切模式不改变输出尺寸", cam.output_size == before, f"{before} 保持不变")


def check_output_size(cam: Camera) -> None:
    print("\n[4] 输出分辨率")

    for want in [(320, 240), (1280, 720), (640, 480)]:
        cam.set_output_size(want)
        frame = wait_frame(cam)
        got = (frame.shape[1], frame.shape[0]) if frame is not None else None
        check(f"set_output_size({want[0]}x{want[1]})",
              got == want, f"read() 返回 {got[0]}x{got[1]}" if got else "取帧失败")

    check("output_size 属性与实际一致", cam.output_size == (640, 480), f"{cam.output_size}")
    check("非法尺寸不崩溃", _no_raise(cam.set_output_size, (0, 0)), "已忽略并告警")


def check_lifecycle() -> None:
    print("\n[5] 生命周期（句柄泄漏)")

    cam = Camera()
    cam.start()
    wait_frame(cam)
    cam.release()
    check("release() 后可重复调用", _no_raise(cam.release) and _no_raise(cam.stop), "无异常")

    # 句柄真的还了 → 能立刻重新占用相机（release 换成 stop 这里就会失败）
    cam2 = Camera()
    ok = _no_raise(cam2.start)
    frame = wait_frame(cam2) if ok else None
    check("释放后可再次 start()", frame is not None, "重新取到帧")
    cam2.release()


# ══════════════════════════════════════════════════════════════════════════════
# 推流层
# ══════════════════════════════════════════════════════════════════════════════

def check_streamer() -> None:
    print("\n[6] MJPEG 推流")

    boom = {"armed": False}
    frame = np.full((120, 160, 3), 64, dtype=np.uint8)

    def provider():
        if boom["armed"]:
            boom["armed"] = False
            raise RuntimeError("注入的 provider 异常")
        return frame

    streamer = MjpegStreamer(
        frame_provider=provider,
        port=PORT,
        custom_routes={"/ping": lambda **kw: {"pong": True}},
    )
    streamer.start()
    time.sleep(1.0)
    base = f"http://127.0.0.1:{PORT}"

    try:
        check("首页 / 返回 200", http_get(base + "/").status == 200, "HTTP 200")
        check("自定义路由 GET", b"pong" in http_get(base + "/ping").read(), "/ping 正常")
        check("自定义路由 POST",
              http_get(base + "/ping", data=b"", method="POST").status == 200, "HTTP 200")

        resp = http_get(base + "/video_feed", timeout=5)
        chunk = resp.read(512)
        check("视频流输出 MJPEG 分段",
              b"--frame" in chunk and b"image/jpeg" in chunk, f"{len(chunk)} 字节")

        boom["armed"] = True
        time.sleep(0.8)
        check("provider 抛异常后流不中断", len(resp.read(512)) > 0, "仍在出帧")
        resp.close()

        t0 = time.perf_counter()
        streamer.stop()
        elapsed = time.perf_counter() - t0
        check("stop() 及时返回", elapsed < 3.0, f"{elapsed:.2f}s")

        time.sleep(0.3)
        freed = False
        try:
            http_get(base + "/", timeout=2)
        except Exception:
            freed = True
        check("stop() 后端口真的释放", freed, f"{PORT} 已关闭")
    finally:
        streamer.stop()


def _no_raise(fn, *a) -> bool:
    try:
        fn(*a)
        return True
    except Exception as e:
        print(f"        ↳ 异常: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 68)
    print("  自检：驱动层 + 推流层")
    print("=" * 68)

    cam = Camera(vflip=False, hflip=False)
    try:
        cam.start()
        check_camera(cam)
        check_params(cam)
        check_sensor_mode(cam)
        check_output_size(cam)
    finally:
        cam.release()

    check_lifecycle()
    check_streamer()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "=" * 68)
    print(f"  结果: {passed}/{len(RESULTS)} 通过, {failed} 失败")
    if failed:
        print("\n  失败项:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}  ({detail})")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
