#!/usr/bin/env python3
"""Display live camera feed on HDMI screen via cage + OpenCV HighGUI.

Runs under cage (Wayland kiosk compositor) to show the camera view in a
fullscreen window on the external HDMI display.

Usage:
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u) WLR_LIBINPUT_NO_DEVICES=1 \\
        cage -- /home/lin/workspace/venv/bin/python tests/test_hdmi_display.py

    # With options:
    sudo XDG_RUNTIME_DIR=/run/user/$(id -u) WLR_LIBINPUT_NO_DEVICES=1 \\
        cage -- /home/lin/workspace/venv/bin/python tests/test_hdmi_display.py \\
            --camera-size 640x480 --exposure 5000

Press 'q' in the window (or Ctrl+C in the terminal) to exit.
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drivers import Camera


DEFAULT_DISPLAY_SIZE = (1024, 600)


def parse_size(s: str):
    parts = s.split("x")
    return int(parts[0]), int(parts[1])


def detect_display_size() -> tuple[int, int] | None:
    """Detect the connected display's preferred resolution via DRM sysfs."""
    modes_glob = sorted(glob.glob("/sys/class/drm/*-HDMI-A-*/modes"))
    if not modes_glob:
        modes_glob = sorted(glob.glob("/sys/class/drm/*/modes"))
    for path in modes_glob:
        try:
            modes = Path(path).read_text().strip().split("\n")
            for mode in modes:
                mode = mode.strip()
                if not mode:
                    continue
                parts = mode.split("x")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
                    if w > 0 and h > 0:
                        return w, h
        except Exception:
            continue
    return None


def build_cage_command(script_args: list[str] | None = None):
    """Print the cage command needed to run this script."""
    import shlex
    venv_python = Path(sys.executable).resolve()
    script = Path(__file__).resolve()
    cmd = (
        f"sudo XDG_RUNTIME_DIR=/run/user/$(id -u) "
        f"WLR_LIBINPUT_NO_DEVICES=1 "
        f"cage -- {venv_python} {script}"
    )
    if script_args:
        cmd += " " + " ".join(shlex.quote(a) for a in script_args)
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Display camera feed on HDMI via cage + OpenCV HighGUI"
    )
    parser.add_argument("--camera-size", default="640x480",
                        help="Camera output size, e.g. 640x480 (default: 640x480)")
    parser.add_argument("--window-size", default=None,
                        help="Display window size, e.g. 1920x1080 (default: auto-detect or 1024x600)")
    parser.add_argument("--no-vflip", action="store_true",
                        help="Disable vertical flip")
    parser.add_argument("--no-hflip", action="store_true",
                        help="Disable horizontal flip")
    parser.add_argument("--exposure", type=int, default=30000,
                        help="Exposure time in µs (default: 30000)")
    parser.add_argument("--gain", type=float, default=3.0,
                        help="Analogue gain (default: 3.0)")
    parser.add_argument("--fullscreen", action="store_true", default=True,
                        help="Start fullscreen (default: True)")
    parser.add_argument("--no-fullscreen", action="store_false", dest="fullscreen",
                        help="Start in a window instead of fullscreen")
    parser.add_argument("--print-command", action="store_true",
                        help="Print the cage command and exit")
    args = parser.parse_args()

    if args.print_command:
        print(build_cage_command(sys.argv[1:]))
        return

    cam_size = parse_size(args.camera_size)

    if args.window_size:
        win_w, win_h = parse_size(args.window_size)
    else:
        detected = detect_display_size()
        if detected:
            win_w, win_h = detected
        else:
            win_w, win_h = DEFAULT_DISPLAY_SIZE

    print(f"Starting camera ({cam_size[0]}x{cam_size[1]}) ...")
    cam = Camera(
        vflip=not args.no_vflip,
        hflip=not args.no_hflip,
        output_size=cam_size,
        exposure_time=args.exposure,
        analogue_gain=args.gain,
    )
    cam.start()

    frame = None
    for _ in range(200):
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        print("ERROR: no camera frame after 10 s — check camera connection")
        cam.release()
        sys.exit(1)

    print(f"Display size: {win_w}x{win_h}")
    print(f"Camera ready — opening {'fullscreen' if args.fullscreen else 'windowed'} display")

    win_name = "Camera HDMI"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win_name, win_w, win_h)

    fps_counter = 0
    fps_start = time.perf_counter()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            display = cv2.resize(frame, (win_w, win_h))
            cv2.imshow(win_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit by 'q' key")
                break

            fps_counter += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 2.0:
                fps = fps_counter / elapsed
                sys.stdout.write(f"\rDisplay FPS: {fps:.1f}   ")
                sys.stdout.flush()
                fps_counter = 0
                fps_start = time.perf_counter()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cv2.destroyAllWindows()
        cam.release()
        print("Done")


if __name__ == "__main__":
    main()
