#!/usr/bin/env python3
"""
Camera Hardware Diagnosis Suite
================================
Diagnoses why a CSI camera is not working on Raspberry Pi 5 + PiSP stack.

Usage:
    source venv/bin/activate
    python tests/test_camera_diagnosis.py

Each test checks one layer of the camera stack and produces a verdict.
"""

import subprocess
import sys
from pathlib import Path


def sh(cmd: str) -> str:
    """Run shell command, return stdout+stderr stripped."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "(timeout)"


# ── Test functions ───────────────────────────────────────────────────────────


def test_kernel_driver() -> tuple[bool, str]:
    """Layer 1: Kernel driver rp1_cfe must be loaded."""
    out = sh("lsmod | grep rp1_cfe || true")
    ok = "rp1_cfe" in out
    detail = "rp1_cfe module IS loaded" if ok else "rp1_cfe module NOT found in lsmod"
    return ok, detail


def test_v4l2_devices() -> tuple[bool, str]:
    """Layer 2: rp1-cfe should expose V4L2 video devices."""
    out = sh("v4l2-ctl --list-devices 2>/dev/null | grep -A5 'rp1-cfe' || true")
    ok = "rp1-cfe" in out and "/dev/video" in out
    lines = out.splitlines() if out else []
    detail = (
        f"{len(lines)} device(s) found: {lines}" if ok
        else "No rp1-cfe V4L2 devices found"
    )
    return ok, detail


def test_dmesg_sensor() -> tuple[bool, str]:
    """Layer 3: dmesg should mention the camera sensor probe."""
    out = sh("dmesg | grep -iE 'ov5647|imx219|imx477|imx708|sensor' | tail -5 || true")
    ok = bool(out)
    return ok, (out if ok else "No sensor-related dmesg messages")


def _i2c_bus_faulted(raw: str) -> bool:
    """Detect if ALL addresses show as occupied (SDA stuck low)."""
    lines = raw.splitlines()[1:]  # skip header
    all_occupied = True
    occupied_count = 0
    for line in lines:
        parts = line.split()[1:]  # skip row number
        for p in parts:
            if p != "--":
                occupied_count += 1
    normal_max = 16  # a single sensor uses at most ~4 addrs; >16 means fault
    return occupied_count > normal_max


def test_i2c_bus() -> tuple[bool, str]:
    """Layer 4: I2C bus should show a sensor at a known address."""
    out = sh("i2cdetect -y 13 2>/dev/null || true")
    if _i2c_bus_faulted(out):
        return False, "I2C bus fault — every address responds (SDA likely stuck low)"
    known_addrs = {"3c": "OV5647", "10": "IMX219", "1a": "IMX477", "3b": "IMX708"}
    found = []
    for addr, name in known_addrs.items():
        if addr in out:
            found.append(f"{name} @ 0x{addr}")
    ok = len(found) > 0
    detail = ", ".join(found) if found else "No known sensor address detected on i2c-13"
    return ok, detail


def test_picamera2() -> tuple[bool, str]:
    """Layer 5: Picamera2 API can enumerate a camera."""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        model = cam.camera_properties.get("Model", "unknown")
        cam.close()
        return True, f"Camera detected: {model}"
    except Exception as e:
        return False, f"Picamera2 error: {e}"


def test_capture() -> tuple[bool, str]:
    """Layer 6: Capture a frame successfully."""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.start()
        frame = cam.capture_array()
        cam.stop()
        cam.close()
        return True, f"Frame captured: shape={frame.shape}, dtype={frame.dtype}"
    except Exception as e:
        return False, f"Capture failed: {e}"


def test_camera_demo() -> tuple[bool, str]:
    """Layer 7: src/camera_demo.py --capture exits 0 and saves a JPEG."""
    samples_dir = Path(__file__).resolve().parent.parent / "samples"
    before = set(samples_dir.glob("capture_test_*.jpg"))
    result = sh(
        f"cd {samples_dir.parent} && {sys.executable} src/camera_demo.py --capture"
        " 2>/dev/null"
    )
    after = set(samples_dir.glob("capture_test_*.jpg"))
    new_files = after - before
    if len(new_files) > 0:
        return True, f"JPEG saved: {list(new_files)[0]}"
    elif "Saved:" in result:
        return True, f"JPEG saved: {result}"
    else:
        return False, f"demo script failed:\n{result}"


# ── Runner ───────────────────────────────────────────────────────────────────


ALL_TESTS = [
    ("Kernel driver (rp1_cfe)", test_kernel_driver),
    ("CSI V4L2 device nodes", test_v4l2_devices),
    ("Sensor in dmesg", test_dmesg_sensor),
    ("I2C sensor address", test_i2c_bus),
    ("Picamera2 API", test_picamera2),
    ("Frame capture", test_capture),
    ("camera_demo.py", test_camera_demo),
]


def run_all() -> None:
    passed = 0
    failed = 0
    print()
    print("=" * 62)
    print("  Camera Hardware Diagnosis")
    print("=" * 62)
    for name, func in ALL_TESTS:
        try:
            ok, detail = func()
        except Exception as e:
            ok, detail = False, str(e)
        status = "  PASS" if ok else "  FAIL"
        print(f"{status}  {name:30s}  {detail}")
        if ok:
            passed += 1
        else:
            failed += 1
    print("=" * 62)
    print(f"  Result: {passed}/{len(ALL_TESTS)} passed, {failed} failed")
    print("=" * 62)

    if failed > 0 and not any(
        t[0] == "Kernel driver (rp1_cfe)" and test_kernel_driver()[0] for t in ALL_TESTS
    ):
        print()
        print("  ROOT CAUSE IDENTIFIED:")
        print("  The rp1_cfe kernel driver is not loaded.")
        print("  The new camera module is NOT detected at the hardware level.")
        print()
        print("  Common causes:")
        print("  1. CSI ribbon cable loose or inserted backwards")
        print("  2. Camera module not compatible with Pi 5 / PiSP stack")
        print("  3. Camera module damaged")
        print("  4. Try: sudo poweroff → reseat cable → power on")
        print()


if __name__ == "__main__":
    run_all()
