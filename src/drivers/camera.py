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
from typing import Any

import cv2
import numpy as np
from picamera2 import Picamera2

logger = logging.getLogger("drivers.camera")

DEFAULT_EXPOSURE_TIME = 20000
DEFAULT_ANALOGUE_GAIN = 1.0
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
        vflip: bool = False,
        hflip: bool = False,
        exposure_time: int = DEFAULT_EXPOSURE_TIME,
        analogue_gain: float = DEFAULT_ANALOGUE_GAIN,
        brightness: float = DEFAULT_BRIGHTNESS,
        contrast: float = DEFAULT_CONTRAST,
        sensor_size: tuple[int, int] | None = None,
    ) -> None:
        self._vflip = vflip
        self._hflip = hflip
        self._exposure_time = exposure_time
        self._analogue_gain = analogue_gain
        self._brightness = brightness
        self._contrast = contrast
        self._sensor_size = sensor_size

        self._cam: Picamera2 | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame: np.ndarray | None = None
        self._sensor_modes: list[dict] = []

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Configure the camera, start streaming, and launch the capture thread."""
        if self._running:
            return

        self._cam = Picamera2()
        if self._sensor_size:
            cfg = self._cam.create_preview_configuration(
                sensor={"output_size": self._sensor_size},
            )
        else:
            cfg = self._cam.create_preview_configuration()
        self._cam.configure(cfg)

        # Cache sensor modes AFTER configure, BEFORE start
        self._sensor_modes = self._cam.sensor_modes

        self._cam.start()

        # Apply initial parameters
        self.set_params({
            "ExposureTime": self._exposure_time,
            "AnalogueGain": self._analogue_gain,
            "Brightness": self._brightness,
            "Contrast": self._contrast,
        })

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera started (thread=%s)", self._thread.name)

    def stop(self) -> None:
        """Stop the capture thread and the camera stream."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cam is not None:
            self._cam.stop()
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

        Thread-safe and non-blocking — returns immediately.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

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

    # ── sensor modes ────────────────────────────────────────────────────────

    @property
    def sensor_modes(self) -> list[dict]:
        """Return the list of available sensor modes (cached at start)."""
        return self._sensor_modes

    def switch_sensor_mode(self, mode_id: int) -> None:
        """Stop, reconfigure with a different sensor mode, and restart.

        Args:
            mode_id: Index into :attr:`sensor_modes`.
        """
        if self._cam is None or mode_id >= len(self._sensor_modes):
            logger.warning("switch_sensor_mode: invalid mode %d", mode_id)
            return

        m = self._sensor_modes[mode_id]

        # Stop thread and camera first
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._cam.stop()

        # Reconfigure
        cfg = self._cam.create_preview_configuration(
            sensor={"output_size": m["size"], "bit_depth": m["bit_depth"]},
        )
        self._cam.configure(cfg)
        self._cam.start()

        # Resume capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Switched to sensor mode %d: %s", mode_id, m["size"])

    # ── internal ───────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Run in daemon thread: continuously capture and cache the latest frame."""
        # Let auto-exposure settle before the first read
        import time
        time.sleep(1.0)

        while self._running:
            try:
                raw = self._cam.capture_array()  # RGBA (H, W, 4)
            except Exception:
                logger.exception("Frame capture failed")
                continue

            bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)

            if self._vflip and self._hflip:
                bgr = cv2.flip(bgr, -1)
            elif self._vflip:
                bgr = cv2.flip(bgr, 0)
            elif self._hflip:
                bgr = cv2.flip(bgr, 1)

            with self._lock:
                self._latest_frame = bgr
