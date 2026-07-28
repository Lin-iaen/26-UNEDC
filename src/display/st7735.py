"""Direct SPI driver for ST7735S 128x160 LCD on Raspberry Pi 5.



Uses spidev (hardware SPI) with kernel CS.



Display is physically rotated 90° (landscape).  The controller runs in

portrait mode (MV=0, WIDTH=128, HEIGHT=160); the physical rotation

swaps the axis mapping naturally.  Renderer produces portrait frames

that appear correctly once the panel is rotated.



MADCTL=0x60 gives MX=1 MY=1 — both axes mirrored, which corrects the

physical mounting orientation for this module.



Pin mapping (BCM):

    SCLK = GPIO11  (SPI0 SCLK — hw)

    MOSI = GPIO10  (SPI0 MOSI — hw)

    CS   = GPIO8   (SPI0 CE0  — hw)

    DC   = GPIO5

    RST  = GPIO22

"""



import logging

import time



import numpy as np

import spidev

import RPi.GPIO as GPIO



logger = logging.getLogger("display.st7735")



WIDTH = 160

HEIGHT = 128



PIN_DC = 5

PIN_RST = 22



CHUNK = 4096





class ST7735:

    """Hardware-SPI ST7735S driver via spidev + RPi.GPIO."""



    def __init__(self) -> None:

        self._spi = spidev.SpiDev()

        self._spi.open(0, 0)

        self._spi.max_speed_hz = 12_500_000

        self._spi.mode = 0



        GPIO.setmode(GPIO.BCM)

        GPIO.setup(PIN_DC, GPIO.OUT)

        GPIO.setup(PIN_RST, GPIO.OUT)



    def _cmd(self, byte: int) -> None:

        GPIO.output(PIN_DC, GPIO.LOW)

        self._spi.xfer2([byte])



    def _data(self, buf: list[int]) -> None:

        GPIO.output(PIN_DC, GPIO.HIGH)

        self._spi.xfer2(buf)



    def _chunked_data(self, buf: list[int]) -> None:

        total = len(buf)

        off = 0

        while off < total:

            end = min(off + CHUNK, total)

            self._data(buf[off:end])

            off = end



    def init(self) -> None:

        GPIO.output(PIN_RST, GPIO.HIGH)

        time.sleep(0.010)

        GPIO.output(PIN_RST, GPIO.LOW)

        time.sleep(0.020)

        GPIO.output(PIN_RST, GPIO.HIGH)

        time.sleep(0.020)



        self._cmd(0x01); time.sleep(0.150)

        self._cmd(0x11); time.sleep(0.500)



        self._cmd(0xB1); self._data([0x01, 0x2C, 0x2D])

        self._cmd(0xB2); self._data([0x01, 0x2C, 0x2D])

        self._cmd(0xB3); self._data([0x01, 0x2C, 0x2D, 0x01, 0x2C, 0x2D])

        self._cmd(0xB4); self._data([0x07])

        self._cmd(0xC0); self._data([0xA2, 0x02, 0x84])

        self._cmd(0xC1); self._data([0xC5])

        self._cmd(0xC2); self._data([0x0A, 0x00])

        self._cmd(0xC3); self._data([0x8A, 0x2A])

        self._cmd(0xC4); self._data([0x8A, 0xEE])

        self._cmd(0xC5); self._data([0x0E])

        self._cmd(0x36); self._data([0x60])

        self._cmd(0x3A); self._data([0x05])



        self._cmd(0xE0)

        self._data([0x0F, 0x1A, 0x0F, 0x18, 0x2F, 0x28, 0x20, 0x22,

                    0x1F, 0x1B, 0x23, 0x37, 0x00, 0x07, 0x02, 0x10])

        self._cmd(0xE1)

        self._data([0x0F, 0x1B, 0x0F, 0x17, 0x33, 0x2C, 0x29, 0x2E,

                    0x30, 0x30, 0x39, 0x3F, 0x00, 0x07, 0x03, 0x10])



        self._cmd(0x13)

        self._cmd(0x29); time.sleep(0.100)

        logger.info("ST7735S init: %dx%d RGB565", WIDTH, HEIGHT)



    def _set_addr(self, x0, y0, x1, y1) -> None:

        self._cmd(0x2A)

        self._data([0x00, x0, 0x00, x1])

        self._cmd(0x2B)

        self._data([0x00, y0, 0x00, y1])



    def fill(self, color: int) -> None:

        self._set_addr(0, 0, WIDTH - 1, HEIGHT - 1)

        hi, lo = color >> 8, color & 0xFF

        line = [hi, lo] * WIDTH

        self._cmd(0x2C)

        for _ in range(HEIGHT):

            self._data(line)



    def display_bgr(self, bgr: np.ndarray) -> None:

        h, w = bgr.shape[:2]

        self._set_addr(0, 0, w - 1, h - 1)



        r = bgr[:, :, 2].ravel().astype(np.uint16)

        g = bgr[:, :, 1].ravel().astype(np.uint16)

        b = bgr[:, :, 0].ravel().astype(np.uint16)

        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | ((b & 0xF8) >> 3)



        data = np.empty(len(rgb565) * 2, dtype=np.uint8)

        data[0::2] = (rgb565 >> 8).astype(np.uint8)

        data[1::2] = (rgb565 & 0xFF).astype(np.uint8)



        self._cmd(0x2C)

        self._chunked_data(data.tolist())



    def clear(self) -> None:

        self.fill(0x0000)



    def cleanup(self) -> None:

        try:

            self._spi.close()

        except Exception:

            pass

        try:

            GPIO.cleanup()

        except Exception:

            pass

        logger.info("ST7735S released")





class LcdDisplay:

    """High-level display interface."""



    def __init__(self) -> None:

        self._dev = ST7735()

        self._dev.init()



    @property

    def width(self) -> int:

        return WIDTH



    @property

    def height(self) -> int:

        return HEIGHT



    def display(self, image: 'Image.Image') -> None:

        arr = np.asarray(image.convert("RGB"))

        self._dev.display_bgr(arr[:, :, ::-1])



    def display_numpy(self, rgb: np.ndarray) -> None:

        self._dev.display_bgr(rgb[:, :, ::-1])



    def display_bgr(self, bgr: np.ndarray) -> None:

        self._dev.display_bgr(bgr)



    def clear(self) -> None:

        self._dev.clear()



    def cleanup(self) -> None:

        self._dev.cleanup() 

