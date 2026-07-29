#!/usr/bin/env python3
"""摄像头画面 → HDMI 输出 → USB 采集卡 → 回环录制视频

流程:
  1. ffmpeg 从 UVC 采集卡录制视频（同时此流唤醒 EDID，使 HDMI 连接）
  2. drm_fb_test 设置 CRTC 让 HDMI 输出有效画面
  3. CSI camera 捕获画面 → 通过 pipe 送入 drm_fb_test → 写入 dumb buffer
  4. 画面经 HDMI → 采集卡 → ffmpeg 保存为视频文件

用法:
    source venv/bin/activate
    python tests/record_hdmi_loopback.py [--duration 10] [--size 640x480]
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drivers import Camera

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRM_FB_SRC = PROJECT_ROOT / "tests" / "drm_fb_test.c"
DRM_FB_BIN = PROJECT_ROOT / "tests" / "drm_fb_test"
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def compile_drm_fb_test() -> bool:
    if DRM_FB_BIN.exists():
        return True
    r = subprocess.run(
        ["gcc", "-o", str(DRM_FB_BIN), str(DRM_FB_SRC),
         "-ldrm", "-I/usr/include/libdrm", "-I/usr/include/drm"],
        capture_output=True, timeout=30,
    )
    return r.returncode == 0


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


def hdmi_connected() -> bool:
    for p in sorted(Path("/sys/class/drm").glob("card*-HDMI-A-*/status")):
        try:
            if p.read_text().strip() == "connected":
                return True
        except Exception:
            continue
    return False


def wait_hdmi_connect(timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if hdmi_connected():
            return True
        time.sleep(0.3)
    return False


def parse_size(s: str) -> tuple[int, int]:
    parts = s.split("x")
    return int(parts[0]), int(parts[1])


def run(args: argparse.Namespace) -> int:
    duration = args.duration
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = SAMPLES_DIR / out_path.name
    video_w, video_h = parse_size(args.size)

    # ── 1. 编译 drm_fb_test ──
    print("[1/5] 编译 drm_fb_test ...")
    if not compile_drm_fb_test():
        print("ERROR: failed to compile")
        return 1

    # ── 2. 查找 UVC 设备 ──
    print("[2/5] 查找 UVC 设备 ...")
    uvc_dev = find_uvc_device()
    if not uvc_dev:
        print("ERROR: UVC device not found")
        return 1
    print(f"      UVC: {uvc_dev}")

    # ── 3. 启动录制 ffmpeg（同时唤醒 HDMI） ──
    print(f"[3/5] 启动录制 {duration}s → {out_path.name} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-y",
         "-f", "v4l2", "-input_format", "mjpeg",
         "-video_size", f"{video_w}x{video_h}",
         "-i", uvc_dev,
         "-t", str(duration),
         "-c", "libx264", "-preset", "ultrafast",
         "-crf", "18", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart",
         str(out_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    if not wait_hdmi_connect():
        print("WARNING: HDMI did not become connected")

    # ── 4. 启动摄像头 ──
    print(f"[4/5] 启动摄像头 (output_size={video_w}x{video_h}) ...")
    cam = Camera(output_size=(video_w, video_h))
    cam.start()
    frame = None
    for _ in range(200):
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        print("ERROR: camera not producing frames")
        cam.release()
        rec.terminate()
        return 1
    actual_h, actual_w = frame.shape[:2]
    print(f"      摄像头实际输出: {actual_w}x{actual_h}")

    # ── 5. 清理旧 drm_fb_test + 启动 stream 模式 ──
    print("[5/5] 启动 drm_fb_test (stream 模式) ...")
    subprocess.run(["pkill", "-9", "drm_fb_test"], capture_output=True)
    time.sleep(0.3)

    drm = subprocess.Popen(
        [str(DRM_FB_BIN), "/dev/dri/card1", "35",
         "stream", str(actual_w), str(actual_h)],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)

    if drm.poll() is not None:
        print("ERROR: drm_fb_test exited early")
        cam.release()
        rec.terminate()
        return 1

    # ── 转发摄像头帧到 drm_fb_test ──
    frames_sent = 0
    while rec.poll() is None:
        f = cam.read()
        if f is not None:
            try:
                drm.stdin.write(f.tobytes())
                drm.stdin.flush()
                frames_sent += 1
            except BrokenPipeError:
                break
        else:
            time.sleep(0.005)

    rec.wait()
    print(f"      发送 {frames_sent} 帧到 HDMI, drm_fb_test alive={drm.poll() is None}")

    # ── 清理 ──
    cam.release()
    drm.stdin.close()
    drm.terminate()
    drm.wait()

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"\n录制完成: {out_path.name} ({size_mb:.1f}MB)")
        probe = subprocess.run(
            ["ffprobe", "-hide_banner", str(out_path)],
            capture_output=True, text=True, timeout=10,
        )
        for line in probe.stderr.split("\n"):
            if "Stream" in line or "Duration" in line:
                print(f"  {line.strip()}")
        return 0
    else:
        print("ERROR: output file not created")
        return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HDMI loopback recording")
    p.add_argument("--duration", type=int, default=10)
    p.add_argument("--size", default="640x480")
    p.add_argument("--output", default="loopback_record.mp4")
    args = p.parse_args()
    sys.exit(run(args))
