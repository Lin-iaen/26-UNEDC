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

3. **Environment Awareness** — The Pi runs Raspberry Pi OS **Lite**: no desktop,
   no display server, and no browser on the device itself.  OpenCV is
   headless-only — `cv2.imshow()`, `cv2.waitKey()`, `cv2.destroyAllWindows()`
   are forbidden.  Debug visually via MJPEG stream or save-to-disk, viewed from
   another machine on the LAN.

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

# Main entrypoint (state machine; hardware wiring still stubbed out)
python -m src.main                            # or: python src/main.py

# Quick capture / burst / stream (thin CLI over src.drivers.Camera)
python src/camera_demo.py --capture
python src/camera_demo.py --capture --vflip
python src/camera_demo.py --test 30           # burst + FPS report
python src/camera_demo.py --stream            # MJPEG HTTP on :5000

# Unattended self-check — driver + streamer contracts, no browser (~40 s)
python tests/test_smoke.py                    # 22 checks, exit 0 = all pass

# Camera console: params / FOV / stream, with hardware read-back (web UI)
python tests/test_camera_console.py           # http://<pi>:5000

# Full-featured 13-param tracking + error analysis (web UI)
python tests/test_tracking_test.py            # http://<pi>:5000

# Rectangle detection with dual-panel debug (web UI)
python tests/test_rectangle_detect.py         # http://<pi>:5000

# Hardware layer diagnosis — run when no image comes out at all
python tests/test_camera_diagnosis.py         # 7-layer check, prints root cause
```

Manual test procedure: `tests/README.md`.  Only one web tool at a time — they
all bind port 5000.

## Project Layout

| Path | Purpose |
|---|---|
| `src/` | Application code (entrypoint in `main.py`) |
| `src/drivers/` | Hardware drivers (`Camera`, `BaseCANMotor`) |
| `src/vision/` | Vision + streaming (`MjpegStreamer`, `BaseTracker`) |
| `tests/` | Test tools + `README.md` (manual test procedure) |
| `samples/` | Captured photos (test evidence) |
| `calibration_data/` | Parameter presets (JSON), saved/loaded via web UI |
| `venv/` | Virtual environment |

## Module Contracts

### `src.drivers.Camera`

```python
cam = Camera(vflip=True, hflip=True)   # NB: both default to True
cam.start()                           # configure + daemon capture thread
frame = cam.read()                    # np.ndarray (H,W,3) BGR, or None
n = cam.frame_id                      # int — bumped once per captured frame
md = cam.get_metadata()               # dict — ACTUAL hw state, {} if not streaming
cam.set_params({"ExposureTime": 30000})
cam.switch_sensor_mode(mode_id)       # FOV + max fps   (frame size unchanged)
cam.set_output_size((1280, 720))      # frame size      (FOV unchanged)
modes = cam.sensor_modes              # list[dict] — probed at start time
w, h = cam.output_size                # actual size read() is returning
cam.stop()                            # stop thread + stream (idempotent)
cam.release()                         # stop() + close hardware handle
```

- Thread-safe: `read()` locks only for copying the latest frame, never during `capture_array()`.
- Returns **BGR** format ready for OpenCV.
- Daemon thread runs `_capture_loop` continuously; `read()` returns a `.copy()` of the cached frame.
- `read()` repeats the cached frame until a new one arrives — poll `frame_id` when
  you need to count *distinct* frames (e.g. measuring real capture FPS).
- **Two independent geometry knobs — do not confuse them:**
  - `output_size` / `set_output_size()` → size of the frames `read()` returns
    (main stream).  This is the *only* thing that changes the frame shape.
  - `sensor_size` / `switch_sensor_mode()` → which sensor region is read out,
    i.e. **FOV and max framerate**.  The ISP scales it to `output_size`, so
    switching modes never changes the frame shape — verify it via `ScalerCrop`
    in `get_metadata()`, not by looking at the resolution.
  Changing `output_size` to a non-4:3 aspect (e.g. 16:9) does crop vertically.
- `full_fov=True` (default) pins the sensor to the **fastest** mode that still
  sees the whole frame and is no smaller than `output_size` (IMX219: 1640×1232 @
  81 fps, not the 3280×2464 @ 21 fps one).  Left to itself picamera2 matches the
  sensor mode to the output size, which at 640×480 is the 2× centre crop (~15%
  FOV).  Pass `full_fov=False` for picamera2's native behaviour.
- **Framerate is capped by exposure**, not just by the mode: `FrameDuration >=
  ExposureTime`.  The 30 000 µs default caps you at ~33 fps no matter what mode
  is selected.  Measured on IMX219 at 640×480 full FOV — 30 000 µs → 33 fps,
  5 000 µs → 81 fps.  Shorten the exposure (and raise gain) before blaming the
  sensor mode.
- Always finish with `release()`, not `stop()` — `stop()` leaves the Picamera2
  handle open and the next process cannot acquire the sensor.
- `read()` returns `None` for the first ~1 s after `start()` (the capture loop
  sleeps to let AE settle) — poll until non-`None` rather than sleeping a fixed
  amount.
- `set_params()` requests a value; `get_metadata()` reports what the hardware
  actually did.  Verify controls with the latter — `ExposureTime` and
  `AnalogueGain` read back, but `Brightness`/`Contrast`/`Saturation`/`Sharpness`
  are ISP-side and have no metadata equivalent.
- `get_metadata()` is non-blocking (cached, ~0.01 ms): `_capture_loop` pulls
  pixels and metadata from **one** `capture_request()` per frame.  Never call
  `capture_metadata()` directly — see the buffer-starvation gotcha below.

### `src.vision.MjpegStreamer`

```python
streamer = MjpegStreamer(
    frame_provider=cam.read,           # callable → np.ndarray | None
    port=5000,
    max_fps=30.0,                        # per-client encode ceiling
    jpeg_quality=75,
    custom_template="<html>...</html>",  # optional
    custom_routes={"/set": handler},     # optional — mounted via add_url_rule
)
streamer.start()                       # Flask in daemon thread, non-blocking
streamer.stop()
```

- Zero business logic — pure pixel pipeline.
- JPEG encoding runs **only when a client is connected** to `/video_feed`.
- Werkzeug log level is set to ERROR to avoid console spam.
- Each custom route receives a unique endpoint name derived from its URL path,
  and accepts both GET and POST.
- `stop()` really shuts the HTTP server down (`werkzeug.make_server` +
  `shutdown()`), so the port is free afterwards — `app.run()` cannot do this.
- An exception from `frame_provider`, or a failed JPEG encode, is logged and
  skipped; the stream keeps running instead of dying mid-response.
- `/` and `/video_feed` accept POST as well as GET: a browser reloading a stalled
  page can re-issue the request as POST, and a 405 there reads as "server dead".
- **Have `frame_provider` return `None` when no new frame is ready.**  The encode
  loop is faster than the camera, so a provider that always returns the cached
  frame makes it re-encode the same pixels several times per capture — invisible
  at 640×480, fatal at 1920×1080.  `max_fps` is only the backstop; gate on
  `cam.frame_id` for the real fix.

## Gotchas

- This is a **Raspberry Pi 5** with the **PiSP** camera pipeline. Code that assumes
  legacy `raspistill` / `raspivid` or `bcm2835-v4l2` will not work.
- `picamera2` is not installable via `pip` on this system (PiWheels may time out).
  Use the `.pth` workaround; do not reinstall or modify `pyvenv.cfg`.
- Full sensor resolution varies by module: OV5647 → 2592×1944, IMX219 → 3280×2464.
- **IMX219 ScalerCrop**: Mode 0 (640×480) uses a 2× center crop (~15% FOV), not a
  full-FOV downscale.  For wide-angle at low resolution, keep `full_fov=True` (the
  default) or pick mode 1/3 explicitly and let the ISP scale down.
- **Reading `picamera2.sensor_modes` DESTROYS the current configuration.**  It
  probes by calling `configure()` once per mode, so whatever you configured
  before reading it is thrown away and the camera runs the last probed mode
  instead.  `Camera.start()` therefore probes **before** `configure()`.  Doing it
  the other way round fails silently — the camera still streams, just not with
  the geometry you asked for.  Verified on IMX219: configure mode 0 → probe →
  start yields `ScalerCrop=(0,2,3280,2460)` instead of `(1000,752,1280,960)`.
- **`capture_metadata()` steals frames from the capture thread.**  It queues its
  own request against the same 4-buffer pool, so every concurrent caller costs
  real framerate — measured on IMX219: 1 caller 21→15.7 fps, 4 callers →7.0 fps,
  8 callers →3.0 fps, with the call itself taking 88–328 ms.  A web UI polling
  `/stats` hits a feedback loop: slower capture → longer blocking → polls pile
  up → slower still, which looks like the video running in slow motion while the
  page stops responding.  `Camera` therefore takes pixels *and* metadata from a
  single `capture_request()` and serves `get_metadata()` from cache.  Anything
  reading metadata per-frame must go through `get_metadata()`.
- **Throttle slider `oninput` handlers** in test UIs.  One request per pixel of
  travel saturates the browser's per-host connection pool — the MJPEG stream
  already holds one of those connections open permanently.
- **ExposureTime/AnalogueGain defaults** from `camera_controls` are static
  descriptions — the ISP's auto-exposure modifies them at runtime.  Always read
  `capture_metadata()` for actual values, and seed UI sliders from metadata,
  not from control defaults.
- Scripts under `tests/` **and** `src/` need
  `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` before importing
  `src.*`, because `python <path>/script.py` puts the *script's* directory on
  `sys.path`, not the project root.  Without it `python src/camera_demo.py`
  dies with `ModuleNotFoundError: No module named 'src'`.
- No formatter / linter / typechecker is configured.
- **Web UIs are opened from a PC/phone on the LAN**, never on the Pi (OS Lite has
  no browser), so `fetch()` in the inline JS is fine.  `test_tracking_test.py`
  uses `XMLHttpRequest` instead — a leftover workaround for the Pi's embedded
  Chromium, harmless but no longer required.  If a UI ever hangs on
  `JS loading...`, check the browser console for a real JS error rather than
  assuming `fetch()` is unsupported.
