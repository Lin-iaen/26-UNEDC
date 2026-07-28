"""LCD display subsystem for ST7735S (128x160) on Raspberry Pi 5.

Provides state-machine driven UI with hardware SPI output and
gpiozero button input.  Designed for field deployment — no web
framework, no GUI toolkit, pure OpenCV + NumPy rendering.
"""

from .st7735 import LcdDisplay
from .buttons import ButtonHandler
from .state_machine import UiState, StateMachine
from . import renderer as Renderer  # module — access via Renderer.render_frame()
from .config import load_hsv, save_hsv

__all__ = [
    "LcdDisplay",
    "ButtonHandler",
    "UiState",
    "StateMachine",
    "Renderer",
    "load_hsv",
    "save_hsv",
]
