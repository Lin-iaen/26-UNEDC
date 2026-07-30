from __future__ import annotations

import logging
import time

import serial

logger = logging.getLogger(__name__)


class UartController:
    def __init__(
        self,
        port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00",
        baudrate: int = 115200,
        dtr: bool | None = None,
        rts: bool | None = None,
        open_delay: float = 0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self._dtr = dtr
        self._rts = rts
        self._open_delay = open_delay
        self.serial: serial.Serial | None = None
        self._connect()

    def _connect(self) -> None:
        try:
            self.serial = serial.Serial()
            self.serial.port = self.port
            self.serial.baudrate = self.baudrate
            self.serial.timeout = 0.1
            if self._dtr is not None:
                self.serial.dtr = self._dtr
            if self._rts is not None:
                self.serial.rts = self._rts
            self.serial.open()
            if self._open_delay > 0:
                time.sleep(self._open_delay)
            logger.info("串口 %s 打开成功，波特率: %s", self.port, self.baudrate)
        except serial.SerialException as e:
            logger.error("无法打开串口 %s: %s", self.port, e)
            self.serial = None

    def read(self) -> bytes:
        if self.serial is None or not self.serial.is_open:
            return b""
        try:
            n = self.serial.in_waiting
            return self.serial.read(n) if n > 0 else b""
        except Exception as e:
            logger.error("串口读取异常: %s", e)
            return b""

    def send_raw(self, data: bytes) -> None:
        if self.serial is None or not self.serial.is_open:
            logger.warning("串口未开启，跳过发送")
            return
        try:
            self.serial.write(data)
            logger.debug("UART 发送: [%s]", " ".join(f"{b:02X}" for b in data))
        except Exception as e:
            logger.error("串口发送异常: %s", e)

    def close(self) -> None:
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
            logger.info("串口已关闭")
