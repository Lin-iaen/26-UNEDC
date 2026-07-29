#!/usr/bin/env python3
"""无线 HDMI 诊断工具

诊断流程：
  1. 监测 HDMI 连接状态随时间变化
  2. 测试不同分辨率下的稳定性
  3. 检测 drmModeSetCrtc 是否导致断连
  4. 逐帧分析录制视频

用法：
    source venv/bin/activate
    python tests/diagnose_wireless_hdmi.py

输出：samples/wireless_diagnosis.txt（诊断报告）
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
DRM_FB_BIN = PROJECT_ROOT / "tests" / "drm_fb_test"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def eprint(s):
    print(s, flush=True)


def hdmi_status():
    for p in sorted(Path("/sys/class/drm").glob("card*-HDMI-A-*/status")):
        try:
            s = p.read_text().strip()
            if s != "unknown":
                return f"{p.name}={s}"
        except Exception:
            continue
    return None


def hdmi_modes():
    for p in sorted(Path("/sys/class/drm").glob("card*-HDMI-A-*/modes")):
        try:
            modes = p.read_text().strip().split("\n")
            return [m for m in modes if m]
        except Exception:
            continue
    return []


def find_uvc():
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


def find_drm():
    for p in Path("/sys/class/drm").glob("card*-HDMI-A-*/status"):
        card = p.parent.name
        return f"/dev/dri/{card.replace('-HDMI-A-1', '').replace('-HDMI-A-2', '')}"
    for d in sorted(Path("/dev/dri").glob("card*")):
        return str(d)
    return "/dev/dri/card1"


# ══════════════════════════════════════════════════════════════════════════════

def test1_connection_monitor():
    """测试 1: 监测 HDMI 连接状态 15 秒"""
    eprint("\n[测试 1] HDMI 连接状态监测 (15s)")
    statuses = []
    for i in range(30):
        s = hdmi_status()
        statuses.append(s)
        time.sleep(0.5)

    unique = list(dict.fromkeys(statuses))  # 保持顺序的去重
    changes = sum(1 for i in range(1, len(statuses)) if statuses[i] != statuses[i-1])
    connected = sum(1 for s in statuses if s and "connected" in s)

    eprint(f"  采样: {len(statuses)} 次, 连接态: {connected}/{len(statuses)}")
    eprint(f"  状态变化: {changes} 次")
    eprint(f"  状态序列: {unique}")
    return {"samples": len(statuses), "connected": connected,
            "changes": changes, "unique_states": unique}


def test2_edid_check():
    """测试 2: 检查 EDID 可用性"""
    eprint("\n[测试 2] EDID 检查")

    # 启动 UVC 流后检查 EDID
    device = find_uvc()
    if not device:
        eprint("  ⚠ UVC 设备未找到")
        return {"edid_bytes": 0}

    uvc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-f", "v4l2", "-input_format", "mjpeg",
         "-video_size", "640x480", "-i", device,
         "-t", "10", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(4)

    edid_info = {}
    for p in Path("/sys/class/drm").glob("card*/edid"):
        try:
            data = p.read_bytes()
            edid_info[p.parent.name] = len(data)
            eprint(f"  {p.parent.name}: {len(data)} bytes EDID")
        except Exception as e:
            edid_info[p.parent.name] = 0
            eprint(f"  {p.parent.name}: {e}")

    modes = hdmi_modes()
    if modes:
        eprint(f"  可用显示模式: {len(modes)}")
        for m in modes:
            eprint(f"    {m}")
    else:
        eprint(f"  无可用显示模式")

    uvc.terminate()
    uvc.wait()
    return {"edid": edid_info, "modes": modes}


def test3_crtc_impact():
    """测试 3: drmModeSetCrtc 是否导致断连"""
    eprint("\n[测试 3] drmModeSetCrtc 对连接的影响")

    device = find_uvc()
    if not device:
        return {}

    # 启动 UVC 流
    uvc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-f", "v4l2", "-input_format", "mjpeg",
         "-video_size", "640x480", "-i", device,
         "-t", "15", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(4)

    status_before = hdmi_status()
    eprint(f"  CRTC 设置前: {status_before}")

    # 启动 drm_fb_test (no stream, no hold — 静态测试图案)
    drm = subprocess.Popen(
        [str(DRM_FB_BIN), "/dev/dri/card1", "35"],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    status_after = hdmi_status()
    eprint(f"  CRTC 设置后: {status_after}")

    # 监测 5 秒内的稳定性
    drops = 0
    for i in range(10):
        s = hdmi_status()
        if not s or "disconnected" in s:
            drops += 1
        time.sleep(0.5)

    eprint(f"  5s 内断连次数: {drops}/10")

    drm.terminate()
    drm.wait()
    uvc.terminate()
    uvc.wait()

    return {
        "before": status_before, "after": status_after, "drops_5s": drops,
    }


def test4_resolution_stability():
    """测试 4: 不同分辨率下录制稳定性"""
    eprint("\n[测试 4] 分辨率稳定性测试")

    device = find_uvc()
    if not device:
        return {}

    results = {}
    for res in ["640x480", "1280x720", "1920x1080"]:
        eprint(f"  ── {res} ──")
        out = SAMPLES_DIR / f"diagnose_{res.replace('x', '_')}.mp4"

        rec = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-y",
             "-f", "v4l2", "-input_format", "mjpeg",
             "-video_size", res, "-i", device,
             "-t", "5", "-c", "copy", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # 等 EDID 稳定

        import cv2
        frames_good = 0
        frames_bad = 0
        rec.wait()

        if out.exists() and out.stat().st_size > 1000:
            cap = cv2.VideoCapture(str(out))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in range(min(total, 20)):
                ret, img = cap.read()
                if not ret:
                    break
                v_grad = float(np.abs(np.diff(img.astype(np.int32), axis=0)).mean())
                if v_grad > 0.5:
                    frames_good += 1
                else:
                    frames_bad += 1
            cap.release()
            eprint(f"    帧: {frames_good} normal / {frames_bad} striped (of {frames_good+frames_bad})")
            out.unlink()
        else:
            eprint(f"    录制失败")
        results[res] = {"good": frames_good, "bad": frames_bad}

    return results


# ══════════════════════════════════════════════════════════════════════════════

def main():
    eprint("=" * 60)
    eprint("  无线 HDMI 诊断工具")
    eprint("=" * 60)
    eprint(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 编译 drm_fb_test
    src = PROJECT_ROOT / "tests" / "drm_fb_test.c"
    if not DRM_FB_BIN.exists():
        subprocess.run(
            ["gcc", "-o", str(DRM_FB_BIN), str(src),
             "-ldrm", "-I/usr/include/libdrm", "-I/usr/include/drm"],
            capture_output=True, timeout=30,
        )

    t1 = test1_connection_monitor()
    t2 = test2_edid_check()
    t3 = test3_crtc_impact()
    t4 = test4_resolution_stability()

    # ── 诊断结论 ──
    eprint("\n" + "=" * 60)
    eprint("  诊断结论")
    eprint("=" * 60)

    issues = []

    if t1.get("connected", 0) < t1.get("samples", 1) * 0.5:
        issues.append("HDMI 连接不稳定（连接率 < 50%）")
    if t1.get("changes", 0) > 5:
        issues.append(f"HDMI 状态频繁变化（{t1['changes']} 次/15s）")

    if t3.get("drops_5s", 0) > 3:
        issues.append("drmModeSetCrtc 后 HDMI 频繁断连")

    for res, data in t4.items():
        if data.get("good", 0) == 0:
            issues.append(f"{res}: 全部帧为竖条纹")

    if not issues:
        eprint("  未检测到明显问题")
    else:
        eprint("  发现以下问题:")
        for i, issue in enumerate(issues, 1):
            eprint(f"  {i}. {issue}")

    eprint("")
    if issues:
        eprint("  建议:")
        for issue in issues:
            if "连接不稳定" in issue:
                eprint("    - 缩短无线 HDMI 收发器距离")
                eprint("    - 避免 5GHz WiFi 干扰")
                eprint("    - 尝试 720p@30 降低带宽")
            if "drmModeSetCrtc" in issue:
                eprint("    - 使用 --wireless 参数运行测试（启用 hold 模式）")
            if "竖条纹" in issue:
                eprint("    - 尝试降低分辨率（720p 或 480p）")
                eprint("    - 检查 HDMI 源是否稳定输出")

    eprint("\n报告已保存")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
