from __future__ import annotations

import struct
from dataclasses import dataclass

from .uart import UartController

CAN_ID_BASE = 0x000
CAN_STATUS_BASE = 0x100
MODE_STANDBY = 0
MODE_TORQUE = 1
MODE_SPEED = 2
MODE_POSITION = 3
MODE_CALIBRATE = 4

KP_SCALE = 10.0 / 65535.0
KDKI_SCALE = 1.0 / 255.0

POS_DECODE = 1.0 / 834.4
SPD_DECODE = 1.0 / 131.07
VQ_DECODE = 1.0 / 819.175

FF_PREAMBLE_MIN = 9

STATUS_PAYLOAD_MIN = 6


@dataclass
class MotorFeedback:
    position_rad: float
    speed_rads: float
    vq: float


class PicoFOCMotor:
    def __init__(self, uart: UartController, motor_id: int = 1) -> None:
        self._uart = uart
        self._can_id = CAN_ID_BASE + motor_id
        self._status_can_id = CAN_STATUS_BASE + motor_id
        self._rx_buffer = bytearray()

    @staticmethod
    def parse_feedback_payload(
        can_id: int, payload: bytes, status_can_id: int
    ) -> MotorFeedback | None:
        if can_id != status_can_id:
            return None
        if len(payload) < STATUS_PAYLOAD_MIN:
            return None
        pos_raw = int.from_bytes(payload[0:2], "little", signed=True)
        spd_raw = int.from_bytes(payload[2:4], "little", signed=True)
        vq_raw = int.from_bytes(payload[4:6], "little", signed=True)
        return MotorFeedback(
            position_rad=pos_raw * POS_DECODE,
            speed_rads=spd_raw * SPD_DECODE,
            vq=vq_raw * VQ_DECODE,
        )

    def read_feedback(self) -> MotorFeedback | None:
        raw = self._uart.read()
        if raw:
            self._rx_buffer.extend(raw)

        while len(self._rx_buffer) >= 12:
            ff_idx = self._rx_buffer.find(b"\xff" * FF_PREAMBLE_MIN)
            if ff_idx < 0:
                self._rx_buffer.clear()
                return None
            if ff_idx > 0:
                self._rx_buffer = self._rx_buffer[ff_idx:]
                continue

            end_ff = FF_PREAMBLE_MIN
            while (end_ff < len(self._rx_buffer)
                   and self._rx_buffer[end_ff] == 0xFF):
                end_ff += 1

            if end_ff >= len(self._rx_buffer):
                return None

            header = self._rx_buffer[end_ff]
            if header & 0x0F != 0x0F:
                self._rx_buffer = self._rx_buffer[1:]
                continue

            dlc = (header >> 4) & 0x0F
            if dlc > 8:
                dlc = 8

            id_start = end_ff + 1
            data_start = id_start + 2
            frame_end = data_start + dlc

            if frame_end > len(self._rx_buffer):
                return None

            can_id = (self._rx_buffer[id_start] << 8) | self._rx_buffer[id_start + 1]
            payload = bytes(self._rx_buffer[data_start:frame_end])

            self._rx_buffer = self._rx_buffer[frame_end:]

            fb = self.parse_feedback_payload(can_id, payload, self._status_can_id)
            if fb is not None:
                return fb

        return None

    @staticmethod
    def _encode_kp(kp: float) -> int:
        raw = round(kp / KP_SCALE)
        return max(0, min(65535, raw))

    @staticmethod
    def _encode_kdki(kdki: float) -> int:
        raw = round(kdki / KDKI_SCALE)
        return max(0, min(255, raw))

    def _build_frame(self, mode: int, target: float, kp: float, kdki: float) -> bytes:
        payload = bytearray(8)
        payload[0] = mode & 0xFF
        payload[1:5] = struct.pack("<f", target)
        payload[5:7] = struct.pack("<H", self._encode_kp(kp))
        payload[7] = self._encode_kdki(kdki) & 0xFF

        frame = bytearray()
        frame.extend(self._can_id.to_bytes(2, "big"))
        frame.extend(payload)
        return bytes(frame)

    def _send_frame(self, mode: int, target: float, kp: float, kdki: float) -> None:
        self._uart.send_raw(self._build_frame(mode, target, kp, kdki))

    def standby(self) -> None:
        self._send_frame(MODE_STANDBY, 0.0, 0.0, 0.0)

    def set_speed(self, target_rad_s: float, kp: float = 0.01, ki: float = 0.004) -> None:
        self._send_frame(MODE_SPEED, target_rad_s, kp, ki)

    def set_position(self, target_rad: float, kp: float = 5.0, kd: float = 0.0) -> None:
        self._send_frame(MODE_POSITION, target_rad, kp, kd)

    def calibrate(self) -> None:
        self._send_frame(MODE_CALIBRATE, 0.0, 5.0, 0.0)

    def stop(self) -> None:
        self.standby()
