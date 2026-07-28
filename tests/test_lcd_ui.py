#!/usr/bin/env python3
"""Standalone LCD + buttons test for ST7735S (128x160).

No camera dependency — runs the full state machine with simulated
frames (color bars).  Validates hardware wiring, SPI communication,
and button event handling.

Usage::

    source venv/bin/activate
    python tests/test_lcd_ui.py          # run with default pins
    python tests/test_lcd_ui.py --no-lcd # render only, skip SPI init
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

MIN_PUSH_INTERVAL = 0.090    # ~11 FPS cap — avoids visible page tearing
TEST_DURATION = 20.0         # seconds before auto-exit
TICK_SLEEP = 0.03            # 30 ms poll interval


def make_color_bar_frame(w: int = 128, h: int = 112) -> np.ndarray:
    bar_w = w // 6
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        x0 = i * bar_w
        x1 = x0 + bar_w
        canvas[:, x0:x1] = color
    return canvas


def push_if_needed(lcd, bgr: np.ndarray, last_push: float,
                   force: bool = False) -> float:
    """Push frame to LCD if enough time has passed since last push.

    Returns the new timestamp (whether pushed or not).
    """
    now = time.monotonic()
    if not force and now - last_push < MIN_PUSH_INTERVAL:
        return last_push

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lcd.display_numpy(rgb)
    return now


def run_test(no_lcd: bool = False) -> None:
    from src.display import (
        ButtonHandler,
        UiState,
        StateMachine,
        Renderer,
        load_hsv,
    )

    print("=== ST7735S LCD UI Test (simplified) ===\n")

    lcd = None
    if not no_lcd:
        try:
            from src.display import LcdDisplay
            lcd = LcdDisplay()
            lcd.clear()
            print("[OK] LCD initialized (ST7735S on SPI0)")
        except Exception as e:
            print(f"[WARN] LCD init failed: {e}")
            no_lcd = True

    buttons = None
    try:
        buttons = ButtonHandler()
        buttons.start()
        print("[OK] Buttons initialized (A=GPIO6, B=GPIO13, C=GPIO19)")
    except Exception as e:
        print(f"[WARN] Button init failed: {e}")

    initial_hsv = load_hsv()
    sm = StateMachine(initial_hsv=initial_hsv)
    print(f"[OK] State machine initialized (HSV: {sm.param_values})")
    print(f"\n  Test runs for {TEST_DURATION:.0f}s — press buttons to try states.\n")

    test_bar = make_color_bar_frame(128, 128)
    states_visited = set()
    tick_count = 0
    max_ticks = int(TEST_DURATION / TICK_SLEEP)
    last_push = 0.0
    last_state = None

    try:
        while tick_count < max_ticks:
            tick_count += 1

            if buttons is not None:
                event = buttons.poll()
                if event is not None:
                    print(f"  [{time.monotonic():.1f}s] {event.name}")
                    sm.handle_event(event)

                    if sm.save_requested:
                        from src.display.config import save_hsv
                        save_hsv(sm.param_values)
                        sm.save_requested = False
                        print("  [OK] HSV params saved")

            # Detect state change — force push
            state_changed = sm.current_state != last_state
            last_state = sm.current_state
            states_visited.add(sm.current_state.name)

            # Render
            camera_frame = None
            if sm.current_state == UiState.DEBUG:
                camera_frame = test_bar

            bgr = Renderer.render_frame(
                state=sm.current_state,
                menu_index=sm.menu_index,
                camera_frame=camera_frame,
                param_values=sm.param_values,
                param_index=sm.param_index,
                edit_mode=sm.edit_mode,
            )

            if lcd is not None:
                force = state_changed or sm.dirty
                last_push = push_if_needed(lcd, bgr, last_push, force=force)
                sm.dirty = False

            time.sleep(TICK_SLEEP)

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    print(f"\n=== Test Results ===")
    print(f"  States visited: {', '.join(sorted(states_visited))}")
    print(f"  HSV params: {sm.param_values}")
    print(f"  Final state: {sm.current_state.name}")
    print()

    if buttons is not None:
        buttons.stop()
    if lcd is not None:
        lcd.cleanup()

    print("[OK] Cleanup done\n")


if __name__ == "__main__":
    no_lcd = "--no-lcd" in sys.argv
    run_test(no_lcd=no_lcd)
