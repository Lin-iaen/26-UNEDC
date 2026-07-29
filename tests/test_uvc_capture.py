#!/usr/bin/env python3
"""USB 视频采集卡 (UVC) 自环采集测试

测试拓扑：Pi HDMI 输出 → HDMI 线 → USB 采集卡 → USB 线 → Pi 自身 USB 口

原理：
  Macrosilicon MS2109 类采集卡在 UVC 推流启动后才暴露 EDID，因此：
  1. 先启动后台 UVC 流 → HDMI 显示 connected
  2. 运行 drm_fb_test → 分配 dumb buffer + drmModeSetCrtc 让 HDMI 输出彩条图案
  3. 采集卡接收 HDMI 信号 → ffmpeg 捕获帧
  4. 验证画面完整性

用法：
    source venv/bin/activate
    python tests/test_uvc_capture.py

输出文件（均保存到 samples/）：
    uvc_loopback.jpg   (MJPEG 1920×1080)
    uvc_loopback_yuyv.png   (YUYV 1920×1080)
    uvc_loopback_640.jpg   (MJPEG 640×480)

依赖：libdrm-dev, gcc (测试脚本自动编译 drm_fb_test.c)

退出码：0=全通过, 1=有失败
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

DRM_FB_SRC = Path(__file__).resolve().parent / "drm_fb_test.c"
DRM_FB_BIN = Path(__file__).resolve().parent / "drm_fb_test"

RESULTS: list[tuple[str, bool, str]] = []
CAPTURE_PATHS: list[Path] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s}  {detail}")
    return ok


def find_uvc_device() -> str | None:
    for dev in sorted(Path("/dev").glob("video*")):
        try:
            r = subprocess.run(
                ["v4l2-ctl", "-d", str(dev), "--info"],
                capture_output=True, text=True, timeout=3,
            )
            if "UVC Camera" in r.stdout:
                return str(dev)
        except Exception:
            continue
    return None


def hdmi_status() -> str | None:
    for p in sorted(Path("/sys/class/drm").glob("card*-HDMI-A-*/status")):
        try:
            s = p.read_text().strip()
            if s != "unknown":
                return f"{p.name}={s}"
        except Exception:
            continue
    return None


def compile_drm_fb_test() -> bool:
    if DRM_FB_BIN.exists():
        return True
    if not DRM_FB_SRC.exists():
        return False
    r = subprocess.run(
        ["gcc", "-o", str(DRM_FB_BIN), str(DRM_FB_SRC),
         "-ldrm", "-I/usr/include/libdrm", "-I/usr/include/drm"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode == 0


def wake_hdmi_and_set_mode(device: str, timeout: float = 10.0) -> tuple[bool, str, subprocess.Popen | None]:
    """启动后台 UVC 流 → 等待 HDMI connected → 运行 drm_fb_test 设模式"""
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-f", "v4l2", "-input_format", "mjpeg",
         "-video_size", "640x480", "-i", device,
         "-t", str(timeout + 10), "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + timeout
    connected = False
    while time.time() < deadline:
        s = hdmi_status()
        if s and "connected" in s:
            connected = True
            break
        time.sleep(0.3)

    drm_proc = None
    if connected:
        drm_proc = subprocess.Popen(
            [str(DRM_FB_BIN), "/dev/dri/card1", "35"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

    return connected, hdmi_status() or "unknown", drm_proc


def ffmpeg_capture(device: str, fmt: str, width: int, height: int,
                   outpath: Path, timeout: float = 15.0) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner",
        "-f", "v4l2",
        "-input_format", fmt.lower(),
        "-video_size", f"{width}x{height}",
        "-i", device,
        "-vframes", "1",
        "-update", "1",
        "-y", str(outpath),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = outpath.stat().st_size > 100
        if ok:
            CAPTURE_PATHS.append(outpath)
        return ok
    except Exception as e:
        print(f"        ↳ ffmpeg error: {e}")
        return False


def validate_frame(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "文件不存在"
    size = path.stat().st_size
    if size < 100:
        return False, f"文件太小 ({size}B)"

    img = cv2.imread(str(path))
    if img is None:
        return False, "OpenCV 无法解码"

    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return False, f"分辨率异常 {w}x{h}"

    mean_std = float(img.std())
    v_grad = float(np.abs(np.diff(img.astype(np.int32), axis=0)).mean())
    unique_rows = int(np.unique(img, axis=0).shape[0])

    summary = f"{w}x{h} {size/1024:.0f}KB std={mean_std:.1f}"

    if mean_std < 5.0:
        return False, f"纯色死图 (std={mean_std:.1f})"

    stripe_ratio = unique_rows / h
    if v_grad < 0.5 and stripe_ratio < 0.05:
        return True, f"竖条纹 (HDMI 无有效信号, unique_rows={unique_rows}/{h})"

    return True, summary


def test_capture(label: str, fmt: str, w: int, h: int, device: str) -> None:
    fname = f"uvc_loopback_{fmt}_{w}x{h}"
    ext = ".jpg" if fmt == "mjpeg" else ".png"
    path = SAMPLES_DIR / (fname + ext)

    ok = ffmpeg_capture(device, fmt, w, h, path)
    check(f"采集 {label}", ok, str(path))
    if ok:
        ok2, detail = validate_frame(path)
        check(f"画面验证 {label}", ok2, detail)


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 66)
    print("  UVC 采集卡 — HDMI 自环采集测试")
    print("  拓扑: Pi HDMI → USB 采集卡 → Pi USB → ffmpeg 捕获")
    print("=" * 66)

    # ── 1. 编译 drm_fb_test ──
    print("\n[0] 编译 drm_fb_test")
    if not check("gcc + libdrm 可用", compile_drm_fb_test(), str(DRM_FB_BIN)):
        print("      需要安装: sudo apt install libdrm-dev gcc")
        return 1

    # ── 2. 定位 UVC 设备 ──
    print("\n[1] 查找 UVC 设备")
    device = find_uvc_device()
    if not check("UVC 采集卡", device is not None, str(device or "无")):
        print("\n  未找到 UVC 设备，请检查 USB 连接")
        return 1

    # ── 3. 唤醒 HDMI + 设置显示模式 ──
    print("\n[2] 唤醒 HDMI + 设置 DRM 模式")
    print("      启动 UVC 流 → 等待 EDID → drm_fb_test → drmModeSetCrtc")
    connected, status, drm_proc = wake_hdmi_and_set_mode(device)
    check("HDMI 已连接 + 模式已设", connected, status)

    if not drm_proc:
        print("      ⚠ drm_fb_test 未运行，后续采集可能得到竖条纹")
    elif drm_proc.poll() is not None:
        print(f"      ⚠ drm_fb_test 已退出 (code={drm_proc.returncode})")

    # ── 4~6. 采集测试 ──
    # 先停掉唤醒用的后台流，释放 /dev/video8
    # (drm_fb_test 保持 HDMI 输出)
    import signal
    subprocess.run(["pkill", "-f", "ffmpeg.*null"], capture_output=True)
    time.sleep(1)

    print("\n[3] MJPEG 1920×1080 采集")
    test_capture("MJPG 1920x1080", "mjpeg", 1920, 1080, device)

    print("\n[4] YUYV 1920×1080 采集 (原始格式)")
    test_capture("YUYV 1920x1080", "yuyv422", 1920, 1080, device)

    print("\n[5] MJPEG 640×480 采集")
    test_capture("MJPG 640x480", "mjpeg", 640, 480, device)

    # ── 7. 清理 ──
    if drm_proc:
        drm_proc.terminate()
        drm_proc.wait(timeout=3)

    # ── 8. 结论 ──
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "=" * 66)
    print(f"  结果: {passed}/{len(RESULTS)} 通过, {failed} 失败")
    if failed:
        print("\n  失败项:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}  ({detail})")
    print(f"\n  输出文件:")
    for p in CAPTURE_PATHS:
        print(f"    samples/{p.name}  ({p.stat().st_size/1024:.0f}KB)")
    print("=" * 66)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
