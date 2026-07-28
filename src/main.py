#!/usr/bin/env python3
"""Entrypoint for the electronics competition main loop.

Dual-mode architecture:
    --mode lab   : Camera + Flask MJPEG stream for LAN tuning (no LCD, no buttons)
    --mode field : Camera + ST7735S LCD + gpiozero buttons (no Flask, default)

Usage:
    python -m src.main --mode lab      # Lab tuning via browser
    python -m src.main --mode field    # Standalone field deployment
    python -m src.main                 # Defaults to field mode
"""

import argparse
import logging
import signal
import sys
import threading
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

# Allow `python src/main.py` as well as `python -m src.main`: in the former,
# sys.path[0] is src/ and the `src` package itself would not be importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drivers import Camera, BaseCANMotor  # noqa: E402 — after sys.path bootstrap
from src.vision import BaseTracker, MjpegStreamer  # noqa: E402

logger = logging.getLogger("main")


class State(Enum):
    INIT = auto()
    IDLE = auto()
    VISION_SEARCH = auto()
    CLOSED_LOOP_TRACKING = auto()
    ERROR = auto()


class ShutdownFlag:
    """Thread-safe flag shared across the module."""

    def __init__(self) -> None:
        self._value = False

    def set(self) -> None:
        self._value = True

    @property
    def is_set(self) -> bool:
        return self._value


def _handle_signal(signum: int, _frame, shutdown: ShutdownFlag) -> None:
    logger.warning("Signal %d received, shutting down ...", signum)
    shutdown.set()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_signal_handlers(shutdown: ShutdownFlag) -> None:
    signal.signal(signal.SIGINT, lambda s, f: _handle_signal(s, f, shutdown))
    signal.signal(signal.SIGTERM, lambda s, f: _handle_signal(s, f, shutdown))


def handle_tracking(state: State, result: dict) -> State:
    """Process tracking result and return the next state.

    Override this with real control logic later.
    """
    return state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Electronics competition main loop with LCD UI"
    )
    parser.add_argument(
        "--mode",
        choices=["lab", "field"],
        default="field",
        help="Operating mode: 'lab' for Flask web stream, 'field' for LCD + buttons (default: field)",
    )
    return parser.parse_args()


# ── Display thread (field mode) ──────────────────────────────────────


def _display_loop(
    shutdown: ShutdownFlag,
    camera: Camera,
) -> None:
    """Background thread: render UI to ST7735S LCD based on state machine.

    Reads camera frames, manages button events, and pushes frames to the
    SPI display.  Runs until shutdown is set.
    """
    from src.display import (
        LcdDisplay,
        ButtonHandler,
        UiState,
        StateMachine,
        Renderer,
        load_hsv,
        save_hsv,
    )

    MIN_PUSH = 0.080    # ~12.5 FPS — avoids visible page tearing on SPI
    TICK_SLEEP = 0.025
    DEBUG_CAMERA_W = 128
    DEBUG_CAMERA_H = 128

    lcd = LcdDisplay()
    buttons = ButtonHandler()
    buttons.start()

    initial_hsv = load_hsv()
    sm = StateMachine(initial_hsv=initial_hsv)
    last_push = 0.0
    last_state = None

    try:
        while not shutdown.is_set:
            event = buttons.poll()
            if event is not None:
                sm.handle_event(event)
                if sm.save_requested:
                    save_hsv(sm.param_values)
                    sm.save_requested = False

            camera_frame_112 = None
            if sm.current_state == UiState.DEBUG:
                raw = camera.read()
                if raw is not None:
                    h, w = raw.shape[:2]
                    side = min(h, w)
                    y0 = (h - side) // 2
                    x0 = (w - side) // 2
                    cropped = raw[y0 : y0 + side, x0 : x0 + side]
                    camera_frame_112 = cv2.resize(
                        cropped, (DEBUG_CAMERA_W, DEBUG_CAMERA_H))

            state_changed = sm.current_state != last_state
            last_state = sm.current_state

            bgr = Renderer.render_frame(
                state=sm.current_state,
                menu_index=sm.menu_index,
                camera_frame=camera_frame_112,
                param_values=sm.param_values,
                param_index=sm.param_index,
                edit_mode=sm.edit_mode,
            )

            force = state_changed or sm.dirty
            now = time.monotonic()
            if force or now - last_push >= MIN_PUSH:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                lcd.display_numpy(rgb)
                last_push = now
                sm.dirty = False

            time.sleep(TICK_SLEEP)

    finally:
        buttons.stop()
        lcd.cleanup()
        logger.info("Display thread exiting")


# ── Lab mode loop ────────────────────────────────────────────────────


def _lab_main(shutdown: ShutdownFlag) -> None:
    """Lab mode: camera + MJPEG streamer for LAN tuning."""
    camera: Camera | None = None
    motor: BaseCANMotor | None = None
    tracker: BaseTracker | None = None

    _latest_annotated: np.ndarray | None = None
    streamer = MjpegStreamer(frame_provider=lambda: _latest_annotated)
    streamer.start()
    logger.info("Lab mode: MjpegStreamer started on port 5000")

    state = State.INIT
    try:
        state = State.IDLE
        logger.info("Lab mode: entering main loop (state=%s)", state.name)

        while not shutdown.is_set:
            if camera is None:
                time.sleep(0.1)
                continue

            try:
                frame = camera.read()
            except Exception:
                logger.exception("Camera read failed")
                state = State.ERROR
                continue

            if frame is None:
                continue

            result: dict = {}
            annotated: np.ndarray = frame

            if tracker is not None:
                try:
                    result, annotated = tracker.process_frame(frame)
                except Exception:
                    logger.exception("Tracker process_frame failed")
                    state = State.ERROR
                    continue

            _latest_annotated = annotated

            try:
                if state == State.IDLE:
                    if tracker is not None:
                        state = State.VISION_SEARCH
                elif state == State.VISION_SEARCH:
                    if result:
                        state = State.CLOSED_LOOP_TRACKING
                elif state == State.CLOSED_LOOP_TRACKING:
                    if not result:
                        state = State.IDLE
                    else:
                        state = handle_tracking(state, result)
                elif state == State.ERROR:
                    state = State.IDLE
            except Exception:
                logger.exception("State machine error")
                state = State.ERROR

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception:
        logger.exception("Unhandled exception in main loop")
    finally:
        logger.info("Shutting down (lab mode) ...")
        if camera is not None:
            try:
                camera.release()
            except Exception:
                logger.exception("Camera release failed")
        if motor is not None:
            try:
                logger.info("Motor disconnected")
            except Exception:
                pass
        try:
            streamer.stop()
            logger.info("Streamer stopped")
        except Exception:
            pass
        logger.info("Lab mode shutdown complete")


# ── Field mode loop ──────────────────────────────────────────────────


def _field_main(shutdown: ShutdownFlag, camera: Camera) -> None:
    """Field mode: vision processing loop (LCD updated by separate thread)."""
    motor: BaseCANMotor | None = None
    tracker: BaseTracker | None = None

    state = State.IDLE
    logger.info("Field mode: entering vision loop (state=%s)", state.name)

    while not shutdown.is_set:
        try:
            frame = camera.read()
        except Exception:
            logger.exception("Camera read failed")
            state = State.ERROR
            continue

        if frame is None:
            continue

        result: dict = {}
        annotated: np.ndarray = frame

        if tracker is not None:
            try:
                result, annotated = tracker.process_frame(frame)
            except Exception:
                logger.exception("Tracker process_frame failed")
                state = State.ERROR
                continue

        try:
            if state == State.IDLE:
                if tracker is not None:
                    state = State.VISION_SEARCH
            elif state == State.VISION_SEARCH:
                if result:
                    state = State.CLOSED_LOOP_TRACKING
            elif state == State.CLOSED_LOOP_TRACKING:
                if not result:
                    state = State.IDLE
                else:
                    state = handle_tracking(state, result)
            elif state == State.ERROR:
                state = State.IDLE
        except Exception:
            logger.exception("State machine error")
            state = State.ERROR


def main() -> None:
    setup_logging()
    args = _parse_args()
    shutdown = ShutdownFlag()
    setup_signal_handlers(shutdown)

    logger.info("Starting in %s mode", args.mode.upper())

    if args.mode == "lab":
        _lab_main(shutdown)
    else:
        camera = Camera()
        camera.start()
        logger.info("Camera started")

        display_thread = threading.Thread(
            target=_display_loop,
            args=(shutdown, camera),
            daemon=True,
            name="lcd-display",
        )
        display_thread.start()
        logger.info("Display thread started (ST7735S on SPI0)")

        try:
            _field_main(shutdown, camera)
        finally:
            shutdown.set()
            display_thread.join(timeout=2.0)
            try:
                camera.release()
                logger.info("Camera released")
            except Exception:
                logger.exception("Camera release failed")


if __name__ == "__main__":
    main()
