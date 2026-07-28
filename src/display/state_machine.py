"""State machine for the LCD user interface.

Four states:
    MENU  — 4-item menu (Debug / Task1 / Task2 / Task3)
    DEBUG — camera feed + HSV cursor/edit overlay
    TASK1, TASK2, TASK3 — no display, just execute task

Button mapping:
    MENU:      A=up, B=down, C=select
    DEBUG:
        Cursor mode:  A/B navigate params, C=enter edit
        Edit mode:    A/B adjust value (short=±1, long=±10), C=confirm, C-long=save+exit
    TASK:      C-long=back to menu
"""

import logging
from enum import Enum, auto
from typing import Optional

from .buttons import ButtonEvent

logger = logging.getLogger("display.state_machine")

MENU_ITEMS = ["Debug", "Task1", "Task2", "Task3"]
HSV_PARAM_NAMES = ["H_min", "S_min", "V_min", "H_max", "S_max", "V_max"]


class UiState(Enum):
    MENU = auto()
    DEBUG = auto()
    TASK1 = auto()
    TASK2 = auto()
    TASK3 = auto()


class StateMachine:
    """Deterministic state machine driven by button events."""

    def __init__(self, initial_hsv: Optional[dict] = None) -> None:
        self.current_state = UiState.MENU
        self.menu_index = 0
        self.param_index = 0
        self.param_values: dict[str, int] = dict(initial_hsv or {})
        self.edit_mode = False
        self.dirty = True
        self.save_requested = False

    def handle_event(self, event: ButtonEvent) -> None:
        handler = {
            UiState.MENU: self._handle_menu,
            UiState.DEBUG: self._handle_debug,
            UiState.TASK1: self._handle_task,
            UiState.TASK2: self._handle_task,
            UiState.TASK3: self._handle_task,
        }[self.current_state]
        handler(event)
        self.dirty = True

    def active_task(self) -> int | None:
        """Return task index (1-3) if in a TASK state, or None."""
        if self.current_state == UiState.TASK1:
            return 1
        if self.current_state == UiState.TASK2:
            return 2
        if self.current_state == UiState.TASK3:
            return 3
        return None

    # ── MENU ──────────────────────────────────────────────────────────

    def _handle_menu(self, event: ButtonEvent) -> None:
        if event in (ButtonEvent.A_SHORT, ButtonEvent.A_LONG):
            self.menu_index = (self.menu_index - 1) % len(MENU_ITEMS)
        elif event in (ButtonEvent.B_SHORT, ButtonEvent.B_LONG):
            self.menu_index = (self.menu_index + 1) % len(MENU_ITEMS)
        elif event == ButtonEvent.C_SHORT:
            self._enter_selected()

    def _enter_selected(self) -> None:
        choice = MENU_ITEMS[self.menu_index]
        logger.info("MENU → %s", choice.upper())
        if choice == "Debug":
            self.current_state = UiState.DEBUG
            self.edit_mode = False
            self.param_index = 0
        elif choice == "Task1":
            self.current_state = UiState.TASK1
        elif choice == "Task2":
            self.current_state = UiState.TASK2
        elif choice == "Task3":
            self.current_state = UiState.TASK3

    # ── DEBUG ─────────────────────────────────────────────────────────

    def _handle_debug(self, event: ButtonEvent) -> None:
        if self.edit_mode:
            self._handle_debug_edit(event)
        else:
            self._handle_debug_cursor(event)

    def _handle_debug_cursor(self, event: ButtonEvent) -> None:
        if event == ButtonEvent.A_SHORT:
            self.param_index = (self.param_index - 1) % len(HSV_PARAM_NAMES)
        elif event == ButtonEvent.B_SHORT:
            self.param_index = (self.param_index + 1) % len(HSV_PARAM_NAMES)
        elif event == ButtonEvent.C_SHORT:
            self.edit_mode = True
            logger.debug("TUNE cursor → edit (%s)", HSV_PARAM_NAMES[self.param_index])
        elif event == ButtonEvent.C_LONG:
            self.current_state = UiState.MENU
            self.save_requested = True
            logger.info("DEBUG → MENU (saved)")

    def _handle_debug_edit(self, event: ButtonEvent) -> None:
        name = HSV_PARAM_NAMES[self.param_index]
        cur = self.param_values.get(name, 128)

        if event == ButtonEvent.A_SHORT:
            self.param_values[name] = max(0, cur - 1)
        elif event == ButtonEvent.A_LONG:
            self.param_values[name] = max(0, cur - 10)
        elif event == ButtonEvent.B_SHORT:
            self.param_values[name] = min(255, cur + 1)
        elif event == ButtonEvent.B_LONG:
            self.param_values[name] = min(255, cur + 10)
        elif event == ButtonEvent.C_SHORT:
            self.edit_mode = False
            logger.debug("TUNE edit → cursor (%s)", name)
        elif event == ButtonEvent.C_LONG:
            self.save_requested = True
            self.edit_mode = False
            self.current_state = UiState.MENU
            logger.info("TUNE edit → MENU (saved)")

    # ── TASK ──────────────────────────────────────────────────────────

    def _handle_task(self, event: ButtonEvent) -> None:
        if event == ButtonEvent.C_LONG:
            self.current_state = UiState.MENU
            logger.info("TASK → MENU")
