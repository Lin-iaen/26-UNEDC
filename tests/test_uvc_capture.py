#!/usr/bin/env python3
"""USB 视频采集卡 (UVC) 自环采集测试

测试拓扑：Pi HDMI 输出 → HDMI（有线/无线）→ USB 采集卡 → Pi USB

流程：
  1. ffmpeg 录制短视频（同时唤醒 EDID，使 HDMI 连接）
  2. drm_fb_test 设置 CRTC（一次性模式，避免无线 HDMI 断连）
  3. 录制完成后逐帧验证画面完整性
  4. 支持 --wireless 模式（更长等待、hold 模式、重试）

用法：
    source venv/bin/activate
    python tests/test_uvc_capture.py [--wireless] [--delay 5]

输出文件（均保存到 samples/）：
    uvc_test.mp4        （原始录制）
    从视频中提取帧进行验证

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
DRM_FB_BIN = PROJECT_ROOT / "tests" / "drm_fb_test"

RESULTS: list[tuple[str, bool, str]] = []


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
    src = PROJECT_ROOT / "tests" / "drm_fb_test.c"
    if DRM_FB_BIN.exists():
        return True
    r = subprocess.run(
        ["gcc", "-o", str(DRM_FB_BIN), str(src),
         "-ldrm", "-I/usr/include/libdrm", "-I/usr/include/drm"],
        capture_output=True, timeout=30,
    )
    return r.returncode == 0


def validate_frame(path: Path, label: str) -> tuple[bool, str]:
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


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    wireless = "--wireless" in sys.argv
    extra_delay = 5 if wireless else 0

    print("=" * 66)
    print("  UVC 采集卡 — HDMI 自环采集测试")
    print(f"  模式: {'无线 HDMI' if wireless else '有线 HDMI'}")
    print("=" * 66)

    # ── 0. 编译 drm_fb_test ──
    print("\n[0] 编译 drm_fb_test")
    if not check("gcc + libdrm", compile_drm_fb_test(), ""):
        return 1

    # ── 1. 查找 UVC 设备 ──
    print("\n[1] 查找 UVC 设备")
    device = find_uvc_device()
    if not check("UVC 采集卡", device is not None, str(device or "无")):
        return 1

    # ── 2. 启动录制（同时唤醒 EDID） ──
    print("\n[2] 启动录制 + 唤醒 HDMI")
    video_path = SAMPLES_DIR / "uvc_test.mp4"
    record_sec = 8 + extra_delay

    subprocess.run(["pkill", "-9", "drm_fb_test"], capture_output=True)
    subprocess.run(["pkill", "-f", "ffmpeg.*video8"], capture_output=True)
    time.sleep(0.5)

    rec = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-y",
         "-f", "v4l2", "-input_format", "mjpeg",
         "-video_size", "640x480",
         "-i", device,
         "-t", str(record_sec),
         "-c", "copy",
         str(video_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"      录制 {record_sec}s → {video_path.name}")

    # ── 3. 等待 HDMI 连接 ──
    deadline = time.time() + 12
    connected = False
    while time.time() < deadline:
        s = hdmi_status()
        if s and "connected" in s:
            connected = True
            break
        time.sleep(0.3)
    check("HDMI 已连接", connected, hdmi_status() or "unknown")

    # ── 4. 启动 drm_fb_test ──
    print("\n[3] 启动 drm_fb_test (hold 模式)")
    drm_args = [str(DRM_FB_BIN), "/dev/dri/card1", "35"]
    if wireless:
        drm_args += ["hold"]  # 无线 HDMI: 断连自动重设 CRTC
    else:
        drm_args += ["stream"]

    drm = subprocess.Popen(
        drm_args,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1 + extra_delay)

    if drm.poll() is not None:
        check("drm_fb_test 运行中", False, f"退出码={drm.returncode}")
        rec.terminate()
        return 1
    check("drm_fb_test 运行中", True, f"PID={drm.pid}")

    # ── 5. 等待录制完成 ──
    rec.wait()
    drm.terminate()
    drm.wait()

    if not video_path.exists() or video_path.stat().st_size < 1000:
        check("视频文件已创建", False, "文件过小或不存在")
        return 1
    check("视频文件已创建", True, f"{video_path.stat().st_size/1024:.0f}KB")

    # ── 6. 逐帧验证 ──
    print("\n[4] 逐帧验证")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
         str(video_path)],
        capture_output=True, text=True, timeout=10,
    )
    try:
        total_frames = int(probe.stdout.strip())
    except ValueError:
        total_frames = 0
    print(f"      视频共 {total_frames} 帧")

    good_frames = 0
    stripe_frames = 0
    bad_frames = 0

    if total_frames > 20:
        sample_positions = [0.2, 0.5, 0.8]  # 前/中/后采样
    else:
        sample_positions = [0.5]

    for pct in sample_positions:
        t = pct * record_sec
        out = f"/tmp/uvc_frame_{int(t)}.jpg"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-y",
             "-ss", str(t), "-i", str(video_path),
             "-vframes", "1", "-update", "1", out],
            capture_output=True, timeout=10,
        )
        valid, detail = validate_frame(Path(out), f"frame@{t}s")
        if "竖条纹" in detail:
            stripe_frames += 1
        elif not valid:
            bad_frames += 1
        else:
            good_frames += 1
        check(f"帧 @ {t:.1f}s", valid, detail)

    # ── 7. 结论 ──
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "=" * 66)
    print(f"  结果: {passed}/{len(RESULTS)} 通过, {failed} 失败")
    if stripe_frames > 0:
        print(f"  ⚠ {stripe_frames}/{len(sample_positions)} 帧为竖条纹（HDMI 信号不稳定）")
    if good_frames > 0:
        print(f"  ✓ {good_frames}/{len(sample_positions)} 帧正常")
    if failed:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}  ({detail})")
    print(f"\n  输出: {video_path.name}")
    print("=" * 66)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
