"""Asynchronous button handler using gpiozero.

Provides a thread-safe event queue polled by the state machine.
Supports both short press (when_pressed) and long press (when_held)
without blocking the main rendering loop.

Wiring (active-low, internal pull-up):
    BtnA (Up/Add)   → GPIO6  → GND
    BtnB (Down/Sub)  → GPIO13 → GND
    BtnC (Select)    → GPIO19 → GND
"""

import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

from gpiozero import Button

logger = logging.getLogger("display.buttons")


class ButtonEvent(Enum):
    """Button event types."""

    A_SHORT = auto()  # BtnA short press
    A_LONG = auto()  # BtnA long press (held)
    B_SHORT = auto()  # BtnB short press
    B_LONG = auto()  # BtnB long press
    C_SHORT = auto()  # BtnC short press
    C_LONG = auto()  # BtnC long press (held)


# Long-press hold time threshold (seconds)
LONG_PRESS_TIME = 0.8

# Default GPIO pins (BCM)
PIN_BTN_A = 6
PIN_BTN_B = 13
PIN_BTN_C = 19


class ButtonHandler:
    """Event-driven button handler with thread-safe queue.

    Creates three gpiozero Buttons with internal pull-up resistors.
    Short press fires on release (before LONG_PRESS_TIME); long press
    fires after holding for LONG_PRESS_TIME seconds.

    Usage::

        handler = ButtonHandler()
        handler.start()
        event = handler.poll()  # non-blocking, returns None if empty
    """

    def __init__(
        self,
        pin_a: int = PIN_BTN_A,
        pin_b: int = PIN_BTN_B,
        pin_c: int = PIN_BTN_C,
        on_event: Optional[Callable[[ButtonEvent], None]] = None,
    ) -> None:
        self._pin_a = pin_a
        self._pin_b = pin_b
        self._pin_c = pin_c
        self._on_event = on_event

        self._event_queue: queue.Queue[ButtonEvent] = queue.Queue()
        self._buttons: list[Button] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        """Initialize GPIO pins and register callbacks."""
        with self._lock:
            if self._running:
                return

            self._held_fired = [False, False, False]

            self._buttons = [
                Button(self._pin_a, pull_up=True, hold_time=LONG_PRESS_TIME),
                Button(self._pin_b, pull_up=True, hold_time=LONG_PRESS_TIME),
                Button(self._pin_c, pull_up=True, hold_time=LONG_PRESS_TIME),
            ]

            events = [
                (ButtonEvent.A_SHORT, ButtonEvent.A_LONG),
                (ButtonEvent.B_SHORT, ButtonEvent.B_LONG),
                (ButtonEvent.C_SHORT, ButtonEvent.C_LONG),
            ]

            for i, (short_ev, long_ev) in enumerate(events):
                btn = self._buttons[i]
                btn.when_pressed = lambda i=i: self._reset_held(i)
                btn.when_held = lambda i=i, e=long_ev: self._fire_held(i, e)
                btn.when_released = lambda i=i, e=short_ev: self._fire_release(i, e)

            self._running = True
            logger.info(
                "Buttons started: A=%d, B=%d, C=%d",
                self._pin_a, self._pin_b, self._pin_c,
            )

    def stop(self) -> None:
        """Release all GPIO resources."""
        with self._lock:
            self._running = False
            for btn in self._buttons:
                try:
                    btn.close()
                except Exception:
                    pass
            self._buttons.clear()
            logger.info("Buttons stopped")

    def poll(self) -> Optional[ButtonEvent]:
        """Non-blocking poll for the next button event, or None."""
        try:
            return self._event_queue.get_nowait()
        except queue.Empty:
            return None

    def flush(self) -> list[ButtonEvent]:
        """Drain all pending events at once."""
        events = []
        while True:
            ev = self.poll()
            if ev is None:
                break
            events.append(ev)
        return events

    def _reset_held(self, i: int) -> None:
        self._held_fired[i] = False

    def _fire_held(self, i: int, ev: ButtonEvent) -> None:
        self._held_fired[i] = True
        self._fire(ev)

    def _fire_release(self, i: int, ev: ButtonEvent) -> None:
        if not self._held_fired[i]:
            self._fire(ev)

    def _fire(self, event: ButtonEvent) -> None:
        """Enqueue an event and invoke the optional callback."""
        if not self._running:
            return
        self._event_queue.put(event)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                logger.exception("Button callback error")
        logger.debug("Button event: %s", event.name)
