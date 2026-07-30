"""Picamera2 hardware wrapper for Raspberry Pi CSI cameras.

Provides a thread-safe, non-blocking Camera class.  No Flask, no HTML, no
web coupling — pure driver layer.

Usage::

    cam = Camera()
    cam.start()
    frame = cam.read()          # BGR ndarray, or None if no frame yet
    cam.set_params({"ExposureTime": 30000})
    cam.release()
"""

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np
from picamera2 import Picamera2

logger = logging.getLogger("drivers.camera")

DEFAULT_EXPOSURE_TIME = 5000
DEFAULT_ANALOGUE_GAIN = 3.0
DEFAULT_BRIGHTNESS = 0.0
DEFAULT_CONTRAST = 1.0


class Camera:
    """Thread-safe Picamera2 wrapper.

    Captures frames in a background daemon thread so that :meth:`read` never
    blocks waiting for the sensor.  Returns BGR-format images ready for OpenCV
    processing.
    """

    def __init__(
        self,
        vflip: bool = True,
        hflip: bool = True,
        exposure_time: int = DEFAULT_EXPOSURE_TIME,
        analogue_gain: float = DEFAULT_ANALOGUE_GAIN,
        brightness: float = DEFAULT_BRIGHTNESS,
        contrast: float = DEFAULT_CONTRAST,
        sensor_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
        full_fov: bool = True,
    ) -> None:
        self._vflip = vflip
        self._hflip = hflip
        self._exposure_time = exposure_time
        self._analogue_gain = analogue_gain
        self._brightness = brightness
        self._contrast = contrast
        # Two INDEPENDENT knobs, easily confused:
        #   sensor_size → which sensor mode to read out (governs FOV and max fps)
        #   output_size → size of the frames read() hands back (ISP scales to it)
        # Leaving output_size as None keeps picamera2's 640×480 default, which is
        # why switching sensor modes alone never changes read()'s frame size.
        self._sensor_size = sensor_size
        self._output_size = output_size
        self._full_fov = full_fov
        self._sensor_mode: dict | None = None

        self._cam: Picamera2 | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._streaming = False
        self._latest_frame: np.ndarray | None = None
        self._latest_metadata: dict[str, Any] = {}
        self._frame_id = 0
        self._sensor_modes: list[dict] = []

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Configure the camera, start streaming, and launch the capture thread."""
        if self._running:
            return

        self._cam = Picamera2()

        # Probe sensor modes BEFORE configuring.  Reading `.sensor_modes`
        # internally configures the camera once per mode, which DISCARDS any
        # configuration already applied — doing it afterwards silently threw
        # away our own config and left the camera running whatever the probe
        # happened to leave behind (verified on IMX219).
        self._sensor_modes = self._cam.sensor_modes

        self._cam.configure(self._build_config())
        self._cam.start()
        self._streaming = True

        # Apply initial parameters
        self._apply_initial_params()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera started (thread=%s)", self._thread.name)

    def stop(self) -> None:
        """Stop the capture thread and the camera stream.  Safe to call twice."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cam is not None and self._streaming:
            self._cam.stop()
            self._streaming = False
        logger.info("Camera stopped")

    def release(self) -> None:
        """Stop the camera and release all hardware resources."""
        self.stop()
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        logger.info("Camera released")

    # ── frame access ───────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Return the most recent BGR frame, or ``None`` if none is available.

        Thread-safe and non-blocking — returns immediately.  The same frame is
        returned repeatedly until the capture thread produces a new one; use
        :attr:`frame_id` to tell them apart.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @property
    def frame_id(self) -> int:
        """Monotonic counter incremented once per captured frame.

        Lets a caller distinguish a genuinely new frame from a repeat of the
        cached one (e.g. when measuring real capture FPS).
        """
        with self._lock:
            return self._frame_id

    def get_metadata(self) -> dict[str, Any]:
        """Return libcamera's metadata for the most recent frame.

        This is the **actual** hardware state (ExposureTime, AnalogueGain, Lux,
        ScalerCrop, …) as opposed to the values requested via :meth:`set_params`
        — use it to verify a control really took effect.  Returns ``{}`` before
        the first frame arrives.

        Non-blocking: the capture thread pulls metadata alongside each frame, so
        this just reads a cached dict.  Do **not** reintroduce a direct
        ``capture_metadata()`` call here — that queues its own request against
        the same small buffer pool the capture thread uses, and starves it.  One
        concurrent caller halves the capture framerate; eight take it from 21 fps
        to 2 fps, which looks exactly like the camera running in slow motion.
        """
        with self._lock:
            return dict(self._latest_metadata)

    # ── dynamic parameters ─────────────────────────────────────────────────

    def set_params(self, params: dict[str, Any]) -> None:
        """Apply one or more libcamera controls at runtime.

        Example::

            cam.set_params({"ExposureTime": 30000, "Brightness": 0.5})
        """
        if self._cam is None:
            logger.warning("set_params called before start() — ignored")
            return
        try:
            self._cam.set_controls(params)
        except Exception:
            logger.exception("set_params failed for keys: %s", list(params.keys()))

    # ── geometry: sensor mode (FOV) and output size (frame size) ───────────

    @property
    def sensor_modes(self) -> list[dict]:
        """Return the list of available sensor modes (cached at start)."""
        return self._sensor_modes

    @property
    def output_size(self) -> tuple[int, int] | None:
        """Size of the frames :meth:`read` returns, as ``(w, h)``.

        ``None`` before the first frame arrives.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            h, w = self._latest_frame.shape[:2]
            return w, h

    def switch_sensor_mode(self, mode_id: int) -> None:
        """Select a different sensor readout mode — changes **FOV and max fps**.

        This does *not* change the size of the frames :meth:`read` returns; the
        ISP scales whatever the sensor produces down to the output size.  Use
        :meth:`set_output_size` for that.

        Args:
            mode_id: Index into :attr:`sensor_modes`.
        """
        if self._cam is None or not 0 <= mode_id < len(self._sensor_modes):
            logger.warning("switch_sensor_mode: invalid mode %d", mode_id)
            return

        self._sensor_mode = self._sensor_modes[mode_id]
        self._reconfigure()
        logger.info("Switched to sensor mode %d: %s", mode_id, self._sensor_mode["size"])

    def set_output_size(self, size: tuple[int, int]) -> None:
        """Resize the main stream — changes the shape of :meth:`read`'s frames.

        Independent of the sensor mode: the ISP scales the sensor readout to
        this size, so FOV is unaffected.  Larger frames cost more CPU to JPEG-
        encode and will lower the achievable stream framerate.

        Args:
            size: ``(width, height)``.  picamera2 may align it slightly; check
                :attr:`output_size` for what was actually granted.
        """
        if self._cam is None:
            logger.warning("set_output_size called before start() — ignored")
            return
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            logger.warning("set_output_size: invalid size %dx%d", w, h)
            return

        self._output_size = (w, h)
        self._reconfigure()
        logger.info("Output size set to %dx%d", w, h)

    # ── internal ───────────────────────────────────────────────────────────

    def _build_config(self) -> dict[str, Any]:
        """Assemble a preview configuration from the current geometry state."""
        # Pin the main-stream format: _capture_loop converts RGBA→BGR and would
        # silently swap R/B if picamera2 ever changed its default.
        main: dict[str, Any] = {"format": "XBGR8888"}
        if self._output_size:
            main["size"] = self._output_size

        kwargs: dict[str, Any] = {"main": main}
        if self._sensor_mode is not None:
            kwargs["sensor"] = {
                "output_size": self._sensor_mode["size"],
                "bit_depth": self._sensor_mode["bit_depth"],
            }
        elif self._sensor_size:
            kwargs["sensor"] = {"output_size": self._sensor_size}
        elif self._full_fov and self._sensor_modes:
            mode = self._pick_full_fov_mode()
            if mode is not None:
                kwargs["sensor"] = {"output_size": mode["size"]}

        return self._cam.create_preview_configuration(**kwargs)

    def _pick_full_fov_mode(self) -> dict | None:
        """Choose the fastest sensor mode that still sees the whole frame.

        Left to itself picamera2 matches the sensor mode to the output size — at
        640×480 that is IMX219's 2× centre crop, i.e. a ~15% field of view.  We
        want the full frame instead, but the *largest* mode is also the slowest
        (3280×2464 caps at 21 fps), so prefer the highest-framerate full-FOV mode
        whose readout is still at least as large as the output — anything
        smaller would be upscaled and lose detail.
        """
        if not self._sensor_modes:
            return None

        widest_area = max(m["crop_limits"][2] * m["crop_limits"][3]
                          for m in self._sensor_modes if "crop_limits" in m) \
            if any("crop_limits" in m for m in self._sensor_modes) else 0

        full = [m for m in self._sensor_modes
                if "crop_limits" in m
                and m["crop_limits"][2] * m["crop_limits"][3] >= widest_area * 0.95]
        if not full:
            return max(self._sensor_modes, key=lambda m: m["size"][0] * m["size"][1])

        need_w, need_h = self._output_size or (0, 0)
        big_enough = [m for m in full
                      if m["size"][0] >= need_w and m["size"][1] >= need_h]
        candidates = big_enough or full
        return max(candidates, key=lambda m: (m.get("fps", 0), -m["size"][0] * m["size"][1]))

    def _reconfigure(self) -> None:
        """Stop stream + thread, apply the current config, restart both.

        Shared by :meth:`switch_sensor_mode` and :meth:`set_output_size` so the
        two knobs compose instead of overwriting each other.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._streaming:
            self._cam.stop()
            self._streaming = False

        # Drop the cached frame and metadata — they belong to the old geometry
        # and would report a ScalerCrop that no longer applies.
        with self._lock:
            self._latest_frame = None
            self._latest_metadata = {}

        self._cam.configure(self._build_config())
        self._cam.start()
        self._streaming = True

        # Note: libcamera controls (exposure, gain, …) survive a
        # stop/configure/start cycle — verified on IMX219 — so runtime values
        # set via set_params() must NOT be re-applied here; doing so would
        # discard whatever the user tuned and snap back to constructor defaults.

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _apply_initial_params(self) -> None:
        """Push the constructor-supplied controls to the sensor."""
        self.set_params({
            "ExposureTime": self._exposure_time,
            "AnalogueGain": self._analogue_gain,
            "Brightness": self._brightness,
            "Contrast": self._contrast,
        })

    def _capture_loop(self) -> None:
        """Run in daemon thread: continuously capture and cache the latest frame."""
        # Let auto-exposure settle before the first read
        time.sleep(1.0)

        while self._running:
            try:
                # One request yields BOTH the pixels and the metadata.  Asking
                # for them separately would queue two requests per frame against
                # a 4-buffer pool and halve the achievable framerate.
                request = self._cam.capture_request()
            except Exception:
                # Back off before retrying: a persistent failure (camera closed,
                # hardware gone) would otherwise spin this thread at 100% CPU
                # and flood the log with tracebacks.
                logger.exception("Frame capture failed")
                time.sleep(0.1)
                continue

            try:
                raw = request.make_array("main")  # RGBA (H, W, 4)
                metadata = request.get_metadata()
                # cvtColor allocates a fresh array, so bgr stays valid after the
                # request (and its buffer) is handed back below.
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
            except Exception:
                logger.exception("Frame conversion failed")
                time.sleep(0.1)
                continue
            finally:
                # Failing to release starves the buffer pool within a few frames.
                request.release()

            if self._vflip and self._hflip:
                bgr = cv2.flip(bgr, -1)
            elif self._vflip:
                bgr = cv2.flip(bgr, 0)
            elif self._hflip:
                bgr = cv2.flip(bgr, 1)

            with self._lock:
                self._latest_frame = bgr
                self._latest_metadata = metadata
                self._frame_id += 1
