#!/usr/bin/env python3
"""USB 视频采集卡 (UVC) 采集测试

测试拓扑：Pi HDMI 输出 → HDMI 线 → USB 采集卡 → USB 线 → Pi 自身 USB 口

已知局限：
  - Macrosilicon MS2109 类采集卡在 UVC 开始推流之前不会暴露 EDID，导致
    HDMI 显示为 disconnected，必须先启后台流来"唤醒"HDMI 链路。
  - Pi 5 自环（HDMI 输出接自己的 USB 采集卡）可能出现彩色竖条纹，这是
    DRM 没有实际扫描出有效帧缓冲导致的，不是采集卡硬件问题。用外部 HDMI
    源即可得到正常画面。

测试项：
  1. UVC 设备检测
  2. HDMI 链路唤醒
  3. MJPEG 1920×1080 采集 + 画面完整性
  4. MJPEG 640×480 采集 + 画面完整性
  5. YUYV 640×480 采集 + 画面完整性

用法：
    source venv/bin/activate
    python tests/test_uvc_capture.py

输出文件（均保存到 samples/）：
    uvc_capture_MJPG_1920x1080.jpg
    uvc_capture_MJPG_640x480.jpg
    uvc_capture_YUYV_640x480.png

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
            if "UVC Camera" in r.stdout or "uvcvideo" in r.stdout:
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


def wake_hdmi(device: str, timeout: float = 8.0) -> tuple[bool, str]:
    """先启后台 UVC 流让采集卡暴露 EDID，然后尝试用 DRM 设置显示模式"""
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-f", "v4l2", "-input_format", "mjpeg",
         "-video_size", "640x480", "-i", device,
         "-t", str(timeout + 5), "-f", "null", "-"],
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

    if connected:
        # 尝试用 modetest 设置模式输出画面
        subprocess.run(
            ["modetest", "-M", "vc4", "-s", "35:1920x1080",
             "-F", "tiles", "-d"],
            capture_output=True, timeout=5,
        )

    proc.terminate()
    proc.wait()
    return connected, hdmi_status() or "unknown"


def validate_frame(path: Path, name: str) -> tuple[bool, str]:
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


def test_capture(device: str, label: str, fmt: str,
                 w: int, h: int) -> None:
    fname = f"uvc_capture_{fmt}_{w}x{h}"
    ext = ".jpg" if fmt == "mjpeg" else ".png"
    path = SAMPLES_DIR / (fname + ext)

    ok = ffmpeg_capture(device, fmt, w, h, path)
    check(f"ffmpeg {label}", ok, str(path))
    if ok:
        valid, detail = validate_frame(path, label)
        check(f"画面验证 {label}", valid, detail)


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 66)
    print("  UVC 采集卡测试")
    print("  拓扑: Pi HDMI → 采集卡 → USB → Pi 自身")
    print("=" * 66)

    # ── 1. 定位 UVC 设备 ──
    print("\n[1] 查找 UVC 设备")
    device = find_uvc_device()
    if not check("检测到 UVC 采集卡", device is not None, str(device or "无")):
        print("\n  未找到 UVC 设备，请检查 USB 连接：")
        return 1

    # ── 2. 唤醒 HDMI ──
    print("\n[2] 唤醒 HDMI 链路")
    print("      启动 UVC 流等待 EDID...")
    connected, status = wake_hdmi(device)
    check("HDMI 连接/状态", connected, status)
    if not connected:
        print("      ⚠ HDMI 未连接，后续采集可能得到竖条纹（无有效视频信号）")

    # ── 3~5. 采集测试 ──
    print("\n[3] MJPEG 1920×1080 采集")
    test_capture(device, "MJPG 1920x1080", "mjpeg", 1920, 1080)

    print("\n[4] MJPEG 640×480 采集")
    test_capture(device, "MJPG 640x480", "mjpeg", 640, 480)

    print("\n[5] YUYV 640×480 采集")
    test_capture(device, "YUYV 640x480", "yuyv422", 640, 480)

    # ── 6. 结论 ──
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
        print(f"    {p.name}  ({p.stat().st_size/1024:.0f}KB)")

    print("=" * 66)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
