#!/usr/bin/env python3
"""PicoFOC 电机控制单元测试 —— 帧构造 + Mock 串口验证，无需硬件。

验证 PicoFOC 协议层：
  - 帧构造（CAN ID + payload）
  - 增益编码 (Kp/KdKi)
  - 全部运行模式 (standby/speed/position/calibrate)
  - Mock UART 集成

用法：
    source venv/bin/activate
    python tests/test_picofoc_motor.py
"""

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drivers.picofoc_motor import PicoFOCMotor
from src.drivers.uart import UartController

MOTOR_ID = 1
EXPECTED_CAN_ID = b"\x00\x01"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s}  {detail}")
    return ok


# ── Mock ──────────────────────────────────────────────────────────────────


class MockUart(UartController):
    def __init__(self):
        self.sent: list[bytes] = []

    def _connect(self) -> None:
        self.serial = None

    def send_raw(self, data: bytes) -> None:
        self.sent.append(data)


# ── 帧构造测试 ────────────────────────────────────────────────────────────


def test_frame_length() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)
    motor.standby()
    check("帧长 10 字节", len(uart.sent[0]) == 10, f"{len(uart.sent[0])}B")


def test_can_id() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=1)
    motor.standby()
    check("CAN ID = 0x0001", uart.sent[0][:2] == b"\x00\x01", uart.sent[0][:2].hex())

    uart2 = MockUart()
    motor2 = PicoFOCMotor(uart2, motor_id=2)
    motor2.standby()
    check("motor_id=2 → CAN ID = 0x0002", uart2.sent[0][:2] == b"\x00\x02", uart2.sent[0][:2].hex())

    uart3 = MockUart()
    motor3 = PicoFOCMotor(uart3, motor_id=0)
    motor3.standby()
    check("motor_id=0 → CAN ID = 0x0000", uart3.sent[0][:2] == b"\x00\x00", uart3.sent[0][:2].hex())


def test_standby() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)
    motor.standby()
    expected = EXPECTED_CAN_ID + b"\x00" * 8
    check("待机帧 payload 全零", uart.sent[0] == expected, uart.sent[0].hex())
    check("待机 mode=0", uart.sent[0][2] == 0, str(uart.sent[0][2]))


def test_speed_mode() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    # speed 5 rad/s, kp=0.01, ki=0.004 (协议文档示例)
    motor.set_speed(5.0, kp=0.01, ki=0.004)
    frame = uart.sent[0]

    check("速度 mode=2", frame[2] == 2, str(frame[2]))

    target = struct.unpack("<f", frame[3:7])[0]
    check("速度 target=5.0 rad/s", abs(target - 5.0) < 1e-6, f"{target:.6f}")

    kp_raw = struct.unpack("<H", frame[7:9])[0]
    kp = kp_raw * (10.0 / 65535.0)
    check("速度 Kp ≈ 0.01", abs(kp - 0.01) < 0.002, f"kp={kp:.5f} (raw={kp_raw})")

    ki_raw = frame[9]
    ki = ki_raw * (1.0 / 255.0)
    check("速度 Ki ≈ 0.004", abs(ki - 0.004) < 0.002, f"ki={ki:.5f} (raw={ki_raw})")


def test_position_mode() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    # position 0°, kp=5.0, kd=0 (协议文档示例)
    motor.set_position(0.0, kp=5.0, kd=0.0)
    frame = uart.sent[0]

    check("位置 mode=3", frame[2] == 3, str(frame[2]))

    target = struct.unpack("<f", frame[3:7])[0]
    check("位置 target=0.0 rad", abs(target) < 1e-6, f"{target:.6f}")

    kp_raw = struct.unpack("<H", frame[7:9])[0]
    kp = kp_raw * (10.0 / 65535.0)
    check("位置 Kp ≈ 5.0", abs(kp - 5.0) < 0.01, f"kp={kp:.4f} (raw={kp_raw})")

    kd_raw = frame[9]
    check("位置 Kd=0", kd_raw == 0, str(kd_raw))


def test_position_45deg() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    angle_rad = math.pi / 4
    motor.set_position(angle_rad, kp=5.0, kd=0.0)
    frame = uart.sent[0]

    target = struct.unpack("<f", frame[3:7])[0]
    check("位置 45° ≈ 0.7854 rad", abs(target - angle_rad) < 1e-4, f"{target:.6f}")


def test_calibrate() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    motor.calibrate()
    frame = uart.sent[0]

    check("校准 mode=4", frame[2] == 4, str(frame[2]))


def test_stop() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    motor.set_speed(5.0)
    motor.stop()
    expected_standby = b"\x00\x01" + b"\x00" * 8
    check("stop 后发送了 2 帧", len(uart.sent) == 2, str(len(uart.sent)))
    check("stop 帧 = standby (全零)", uart.sent[1] == expected_standby,
          f"实际={uart.sent[1].hex()} 期望={expected_standby.hex()}")
    check("stop 帧 mode=0", uart.sent[1][2] == 0, str(uart.sent[1][2]))


# ── 增益编码测试 ──────────────────────────────────────────────────────────


def test_kp_encoding() -> None:
    check("Kp=0 → raw=0", PicoFOCMotor._encode_kp(0.0) == 0, "")
    check("Kp=10 → raw=65535", PicoFOCMotor._encode_kp(10.0) == 65535,
          f"{PicoFOCMotor._encode_kp(10.0)}")
    # 5.0 / (10.0/65535) = 32767.5 → banker's rounding → 32768
    check("Kp=5 → raw=32768", PicoFOCMotor._encode_kp(5.0) == 32768,
          f"{PicoFOCMotor._encode_kp(5.0)}")
    check("Kp=10.1 → 钳位 65535", PicoFOCMotor._encode_kp(10.1) == 65535, "")
    check("Kp=-1 → 钳位 0", PicoFOCMotor._encode_kp(-1.0) == 0, "")


def test_kdki_encoding() -> None:
    check("KdKi=0 → raw=0", PicoFOCMotor._encode_kdki(0.0) == 0, "")
    check("KdKi=1 → raw=255", PicoFOCMotor._encode_kdki(1.0) == 255,
          f"{PicoFOCMotor._encode_kdki(1.0)}")
    check("KdKi=1.5 → 钳位 255", PicoFOCMotor._encode_kdki(1.5) == 255, "")
    check("KdKi=-1 → 钳位 0", PicoFOCMotor._encode_kdki(-1.0) == 0, "")


# ── 协议文档向量验证 ──────────────────────────────────────────────────────


def test_doc_vector_standby() -> None:
    """验证协议文档 §5.1 待机指令。"""
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)
    motor.standby()
    expected = bytes.fromhex("00 01 00 00 00 00 00 00 00 00")
    check("向量-待机", uart.sent[0] == expected, uart.sent[0].hex())


def test_doc_vector_speed() -> None:
    """验证协议文档 §5.1 速度 5 rad/s。"""
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)
    motor.set_speed(5.0, kp=0.01, ki=0.004)
    expected = bytes.fromhex("00 01 02 00 00 A0 40 42 00 01")
    check("向量-速度 5 rad/s", uart.sent[0] == expected,
          f"期望={expected.hex()} 实际={uart.sent[0].hex()}")


def test_doc_vector_position_0() -> None:
    """验证协议文档 §5.1 位置 0°。"""
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)
    motor.set_position(0.0, kp=5.0, kd=0.0)
    expected = bytes.fromhex("00 01 03 00 00 00 00 00 80 00")
    check("向量-位置 0°", uart.sent[0] == expected,
          f"期望={expected.hex()} 实际={uart.sent[0].hex()}")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 62)
    print("  PicoFOC 电机协议单元测试")
    print("=" * 62)

    test_frame_length()
    test_can_id()
    test_standby()
    test_speed_mode()
    test_position_mode()
    test_position_45deg()
    test_calibrate()
    test_stop()
    test_kp_encoding()
    test_kdki_encoding()
    test_doc_vector_standby()
    test_doc_vector_speed()
    test_doc_vector_position_0()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "=" * 62)
    print(f"  结果: {passed}/{len(RESULTS)} 通过, {failed} 失败")
    if failed:
        print("\n  失败项:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}  ({detail})")
    print("=" * 62)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
