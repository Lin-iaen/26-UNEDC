# AGENTS.md

## Architecture Principles

1. **Modular Composition over Inheritance** — Favor simple classes with direct
   method contracts over deep ABC hierarchies.  A class that exposes `start() /
   read() / set_params() / release()` is preferable to an abstract base with
   three partial methods.  ABCs are only warranted when at least three
   interchangeable implementations exist.

2. **Pull-Model Streaming** — Web streamers must receive frames via an injected
   `frame_provider` callable (function, lambda, or bound method), never via a
   push-style `update_frame()` interface.  The provider is called lazily inside
   the MJPEG generator — JPEG encoding only runs when a browser is connected.

3. **Environment Awareness** — OpenCV is headless-only. `cv2.imshow()`,
   `cv2.waitKey()`, `cv2.destroyAllWindows()` are forbidden. Debug visually via
   MJPEG stream or save-to-disk.

4. **Decoupled Design** — Vision processing, control algorithm, and hardware
   driver modules must never import each other directly.  The main program
   wires them together via dependency injection.  Each module depends only on
   the interfaces it consumes (a callable, a dict, a class reference).

## Environment

- **venv**: `./venv`, activate with `source venv/bin/activate`
- **Python**: 3.13
- **OpenCV**: headless — `cv2.imshow()` / `waitKey()` / `destroyAllWindows()` are unavailable
- **picamera2**: installed as Debian system package (not in venv). A `.pth` file at
  `venv/lib/python3.13/site-packages/system_dist.pth` points to
  `/usr/lib/python3/dist-packages` so the venv can import it.
- **Flask**: available in venv for MJPEG streaming.

## Camera

- Platform: Raspberry Pi 5 + PiSP camera stack
- Sensors tested: OV5647 (v1), IMX219 (v2)
- **Do NOT use `cv2.VideoCapture`** to access the CSI camera — the raw V4L2 device
  from `rp1-cfe` driver streams Bayer data that OpenCV cannot decode directly.
  Always use `picamera2.Picamera2` via `src.drivers.Camera`.

## Commands

```bash
source venv/bin/activate

# Quick capture / burst / stream (standalone, no src.drivers)
python src/camera_demo.py --capture
python src/camera_demo.py --capture --vflip
python src/camera_demo.py --test 30           # burst + FPS report
python src/camera_demo.py --stream            # MJPEG HTTP on :5000

# Full-featured 13-param tracking + error analysis (web UI)
python tests/test_tracking_test.py            # http://<pi>:5000

# Rectangle detection with dual-panel debug (web UI)
python tests/test_rectangle_detect.py         # http://<pi>:5000

# AE lock / unlock debug tool
python tests/test_ae_debug.py                 # http://<pi>:5000

# Hardware layer diagnosis
python tests/test_camera_diagnosis.py         # 7-layer check, prints root cause
```

## Project Layout

| Path | Purpose |
|---|---|
| `src/` | Application code (entrypoint in `main.py`) |
| `src/drivers/` | Hardware drivers (`Camera`, `BaseCANMotor`) |
| `src/vision/` | Vision + streaming (`MjpegStreamer`, `BaseTracker`) |
| `tests/` | Test suites and diagnosis tools |
| `samples/` | Captured photos (test evidence) |
| `calibration_data/` | Parameter presets (JSON), saved/loaded via web UI |
| `docs/` | Documentation |
| `venv/` | Virtual environment |

## Module Contracts

### `src.drivers.Camera`

```python
cam = Camera(vflip=False, hflip=False)
cam.start()                           # configure + daemon capture thread
frame = cam.read()                    # np.ndarray (H,W,3) BGR, or None
cam.set_params({"ExposureTime": 30000})
cam.switch_sensor_mode(mode_id)       # stop → reconfigure → restart
modes = cam.sensor_modes              # list[dict] — cached at start time
cam.release()                         # stop thread + close hardware
```

- Thread-safe: `read()` locks only for copying the latest frame, never during `capture_array()`.
- Returns **BGR** format ready for OpenCV.
- Daemon thread runs `_capture_loop` continuously; `read()` returns a `.copy()` of the cached frame.

### `src.vision.MjpegStreamer`

```python
streamer = MjpegStreamer(
    frame_provider=cam.read,           # callable → np.ndarray | None
    port=5000,
    custom_template="<html>...</html>",  # optional
    custom_routes={"/set": handler},     # optional — mounted via add_url_rule
)
streamer.start()                       # Flask in daemon thread, non-blocking
streamer.stop()
```

- Zero business logic — pure pixel pipeline.
- JPEG encoding runs **only when a client is connected** to `/video_feed`.
- Werkzeug log level is set to ERROR to avoid console spam.
- Each custom route receives a unique endpoint name derived from its URL path.

## Gotchas

- This is a **Raspberry Pi 5** with the **PiSP** camera pipeline. Code that assumes
  legacy `raspistill` / `raspivid` or `bcm2835-v4l2` will not work.
- `picamera2` is not installable via `pip` on this system (PiWheels may time out).
  Use the `.pth` workaround; do not reinstall or modify `pyvenv.cfg`.
- Full sensor resolution varies by module: OV5647 → 2592×1944, IMX219 → 3280×2464.
- **IMX219 ScalerCrop**: Mode 0 (640×480) uses a 2× center crop, not full FOV
  downscale.  For wide-angle at low resolution, use Mode 1/3 with ISP scaling.
- **`picamera2.sensor_modes`** internally calls `configure()` and requires the
  camera to be stopped.  `Camera` caches modes after configure and before start
  to avoid this race.
- **ExposureTime/AnalogueGain defaults** from `camera_controls` are static
  descriptions — the ISP's auto-exposure modifies them at runtime.  Always read
  `capture_metadata()` for actual values, and seed UI sliders from metadata,
  not from control defaults.
- Tests in `tests/` need `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
  because they are not inside the `src` package hierarchy.
- No formatter / linter / typechecker is configured.
- **RPi embedded Chromium does not support `fetch()`.** Test web UIs with inline JS
  must use `XMLHttpRequest` (sync for `setParam`, async with `onreadystatechange`
  for save/load/presets). Only `test_tracking_test.py` has been ported so far;
  other test scripts still use `fetch()` and will show `JS loading...` forever.
