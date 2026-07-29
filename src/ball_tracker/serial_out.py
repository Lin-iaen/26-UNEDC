import os
import serial
import serial.tools.list_ports

VID = 0x1a86
PID = 0x7523
BAUDRATE = 115200


def find_ch340() -> str:
    for port in serial.tools.list_ports.comports():
        if port.vid == VID and port.pid == PID:
            return port.device
    by_id = "/dev/serial/by-id"
    if os.path.isdir(by_id):
        for entry in os.listdir(by_id):
            if "1a86" in entry or "USB_Serial" in entry:
                return os.path.join(by_id, entry)
    raise RuntimeError(
        f"CH340 (VID={VID:04x}:PID={PID:04x}) not found. "
        "Check USB connection."
    )


class SerialOut:
    def __init__(self, port: str | None = None):
        if port is None:
            port = find_ch340()
        self._ser = serial.Serial(port, BAUDRATE, timeout=0)

    def send(self, pos_mm: float) -> None:
        self._ser.write(f"{pos_mm:.1f}\n".encode())

    def close(self) -> None:
        self._ser.close()
