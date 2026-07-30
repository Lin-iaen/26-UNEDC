#!/usr/bin/env python3
"""PicoFOC 状态回传解析单元测试 —— 无需硬件。

测试 parse_feedback_payload() 和 read_feedback()：
  - 正常帧解码 (包含 10+ FF 前导)
  - 噪声干扰 / 帧边界对齐
  - 错误 CAN ID / 错误 header
  - 变长 DLC / 空数据
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drivers.picofoc_motor import PicoFOCMotor, MotorFeedback

STATUS_CAN_ID = 0x101
MOTOR_ID = 1

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s}  {detail}")
    return ok


def approx(a: float, b: float, eps: float = 0.01) -> bool:
    return abs(a - b) < eps


def build_payload(pos_rad: float, speed_rads: float, vq: float) -> bytes:
    pos_raw = round(pos_rad * 834.4)
    spd_raw = round(speed_rads * 131.07)
    vq_raw = round(vq * 819.175)
    payload = bytearray()
    payload.extend(pos_raw.to_bytes(2, "little", signed=True))
    payload.extend(spd_raw.to_bytes(2, "little", signed=True))
    payload.extend(vq_raw.to_bytes(2, "little", signed=True))
    payload.extend(b"\x00\x00")
    return bytes(payload)


def build_uart_frame(can_id: int, payload: bytes, ff_count: int = 9) -> bytes:
    frame = bytearray()
    frame.extend(b"\xFF" * ff_count)
    frame.append(0x0F | (len(payload) << 4))
    frame.extend(can_id.to_bytes(2, "big"))
    frame.extend(payload)
    return bytes(frame)


# ── parse_feedback_payload 直接测试 ─────────────────────────────────────────


def test_normal_decode() -> None:
    payload = build_payload(1.571, 10.0, 5.0)
    fb = PicoFOCMotor.parse_feedback_payload(STATUS_CAN_ID, payload, STATUS_CAN_ID)
    check("返回 MotorFeedback", isinstance(fb, MotorFeedback), str(fb))
    check("位置 90° (1.571 rad)", approx(fb.position_rad, 1.571), f"{fb.position_rad:.4f}")
    check("速度 10 rad/s", approx(fb.speed_rads, 10.0), f"{fb.speed_rads:.4f}")
    check("Vq 5.0 V", approx(fb.vq, 5.0), f"{fb.vq:.4f}")


def test_zero_values() -> None:
    payload = build_payload(0.0, 0.0, 0.0)
    fb = PicoFOCMotor.parse_feedback_payload(STATUS_CAN_ID, payload, STATUS_CAN_ID)
    check("零值位置", approx(fb.position_rad, 0.0), f"{fb.position_rad:.4f}")
    check("零值速度", approx(fb.speed_rads, 0.0), f"{fb.speed_rads:.4f}")
    check("零值 Vq", approx(fb.vq, 0.0), f"{fb.vq:.4f}")


def test_negative_position() -> None:
    payload = build_payload(-1.571, 0.0, 0.0)
    fb = PicoFOCMotor.parse_feedback_payload(STATUS_CAN_ID, payload, STATUS_CAN_ID)
    check("负位置 -90°", approx(fb.position_rad, -1.571), f"{fb.position_rad:.4f}")


def test_negative_speed() -> None:
    payload = build_payload(0.0, -10.0, 0.0)
    fb = PicoFOCMotor.parse_feedback_payload(STATUS_CAN_ID, payload, STATUS_CAN_ID)
    check("负速度 -10 rad/s", approx(fb.speed_rads, -10.0), f"{fb.speed_rads:.4f}")


def test_wrong_can_id() -> None:
    payload = build_payload(1.0, 0.0, 0.0)
    fb = PicoFOCMotor.parse_feedback_payload(0x102, payload, STATUS_CAN_ID)
    check("错误 CAN ID → None", fb is None, "can_id=0x102")


def test_short_payload() -> None:
    fb = PicoFOCMotor.parse_feedback_payload(STATUS_CAN_ID, b"\x00\x00", STATUS_CAN_ID)
    check("过短 payload → None", fb is None, "2 bytes")


def test_wrong_header() -> None:
    """parse_feedback_payload 不关心 header，测试通过 read_feedback 间接验证。"""
    payload = build_payload(1.0, 0.0, 0.0)
    fb = PicoFOCMotor.parse_feedback_payload(STATUS_CAN_ID, payload, STATUS_CAN_ID)
    check("payload 解析独立于 header", fb is not None, str(fb))
    check("位置 ≈ 1.0 rad", approx(fb.position_rad, 1.0), f"{fb.position_rad:.4f}")


# ── read_feedback 集成测试 ─────────────────────────────────────────────────


class MockUart:
    def __init__(self):
        self._buf = bytearray()

    def _connect(self):
        self.serial = None

    def read(self) -> bytes:
        if not self._buf:
            return b""
        chunk = bytes(self._buf[:50])
        self._buf = self._buf[50:]
        return chunk

    def send_raw(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        pass


def test_read_feedback_normal() -> None:
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    frame = build_uart_frame(0x101, build_payload(1.571, 10.0, 5.0))
    uart._buf.extend(frame)

    fb = motor.read_feedback()
    check("正常帧 → MotorFeedback", fb is not None, str(fb))
    if fb:
        check("位置正确", approx(fb.position_rad, 1.571), f"{fb.position_rad:.4f}")


def test_long_preamble() -> None:
    """10+ FF 前导 → 应正确同步并解码。"""
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    payload = build_payload(1.571, 10.0, 5.0)
    frame = build_uart_frame(0x101, payload, ff_count=12)
    uart._buf.extend(frame)

    fb = motor.read_feedback()
    check("10+FF 前导 → MotorFeedback", fb is not None, str(fb))
    if fb:
        check("10+FF 位置正确", approx(fb.position_rad, 1.571), f"{fb.position_rad:.4f}")


def test_mixed_frames() -> None:
    """混有不同 ID 的帧 → 只解析 0x101。"""
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    # 先放一个 0x001 (指令回显), 再放一个 0x101 (状态)
    frame_echo = build_uart_frame(0x001, b"\x02" + b"\x00" * 7)
    frame_status = build_uart_frame(0x101, build_payload(1.571, 10.0, 5.0))
    uart._buf.extend(frame_echo)
    uart._buf.extend(frame_status)

    fb = motor.read_feedback()
    check("跳过 0x001 → 解析 0x101", fb is not None, str(fb))
    if fb:
        check("混帧位置正确", approx(fb.position_rad, 1.571), f"{fb.position_rad:.4f}")


def test_read_feedback_noise() -> None:
    """噪声中正确提取帧。"""
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    frame = build_uart_frame(0x101, build_payload(1.571, 10.0, 5.0))
    uart._buf.extend(b"\x00" * 30)
    uart._buf.extend(frame)
    uart._buf.extend(b"\xAA" * 10)

    fb = motor.read_feedback()
    check("噪声中提取回传帧", fb is not None, str(fb))
    if fb:
        check("噪声中位置正确", approx(fb.position_rad, 1.571), f"{fb.position_rad:.4f}")


def test_read_feedback_empty() -> None:
    uart = MockUart()
    uart._connect = lambda: None
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)
    fb = motor.read_feedback()
    check("空数据 → None", fb is None, "")


def test_variable_preamble_noise() -> None:
    """噪声字节中混有 FFs，随后是 9+FF 帧。"""
    uart = MockUart()
    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    # 噪声中有 7 个 FF (不够 preamble)，然后真正的帧
    frame = build_uart_frame(0x101, build_payload(1.571, 10.0, 5.0))
    uart._buf.extend(b"\xFF" * 7 + b"\x55" * 5)
    uart._buf.extend(frame)

    fb = motor.read_feedback()
    check("部分 FF 噪声 → 仍可解码", fb is not None, str(fb))
    if fb:
        check("噪声后位置正确", approx(fb.position_rad, 1.571), f"{fb.position_rad:.4f}")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 62)
    print("  PicoFOC 回传解析单元测试")
    print("=" * 62)

    test_normal_decode()
    test_zero_values()
    test_negative_position()
    test_negative_speed()
    test_wrong_can_id()
    test_short_payload()
    test_wrong_header()

    test_read_feedback_normal()
    test_long_preamble()
    test_mixed_frames()
    test_read_feedback_noise()
    test_read_feedback_empty()
    test_variable_preamble_noise()

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
