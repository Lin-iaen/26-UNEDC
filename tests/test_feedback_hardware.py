#!/usr/bin/env python3
"""电机回传角度测试 —— 发送位置指令，读取 CAN 卡回传的状态帧。

用法:
    source venv/bin/activate
    python tests/test_feedback_hardware.py

流程:
    1. 校准 → 2. 转到 45° 并读回传 → 3. 转到 0° 并读回传 → 4. 待机
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from src.drivers.uart import UartController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
BAUD = 4000000
MOTOR_ID = 1
CAN_ID_CTRL = 0x000 + MOTOR_ID
CAN_ID_STATUS = 0x100 + MOTOR_ID

POS_SCALE = 1.0 / 834.4
SPD_SCALE = 1.0 / 131.07
VQ_SCALE = 1.0 / 819.175


def build_cmd(mode: int, target: float, kp: float, kdki: float) -> bytes:
    import struct
    kp_raw = round(kp / (10.0 / 65535.0))
    kdki_raw = round(kdki / (1.0 / 255.0))
    payload = bytearray(8)
    payload[0] = mode
    payload[1:5] = struct.pack("<f", target)
    payload[5:7] = struct.pack("<H", max(0, min(65535, kp_raw)))
    payload[7] = max(0, min(255, kdki_raw))
    frame = bytearray()
    frame.extend(CAN_ID_CTRL.to_bytes(2, "big"))
    frame.extend(payload)
    return bytes(frame)


def parse_status(raw: bytes):
    """Can卡返回格式: 9×FF + header + 2B ID(big) + 8B payload"""
    if len(raw) < 20:
        return None
    if raw[:9] != b"\xff" * 9:
        return None
    if raw[9] & 0x0F != 0x0F:
        return None
    can_id = (raw[10] << 8) | raw[11]
    if can_id != CAN_ID_STATUS:
        return None
    pos = int.from_bytes(raw[12:14], "little", signed=True) * POS_SCALE
    spd = int.from_bytes(raw[14:16], "little", signed=True) * SPD_SCALE
    vq = int.from_bytes(raw[16:18], "little", signed=True) * VQ_SCALE
    return pos, spd, vq


def read_feedback(uart, timeout=2.0):
    """读取并解析一条回传帧，超时返回 None"""
    buf = bytearray()
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = uart.read()
        if data:
            buf.extend(data)
        while len(buf) >= 20:
            idx = buf.find(b"\xff" * 9)
            if idx < 0:
                buf = buf[-18:]
                break
            if idx + 20 > len(buf):
                break
            fb = parse_status(bytes(buf[idx:idx + 20]))
            buf = buf[idx + 20:]
            if fb is not None:
                return fb
    return None


def main():
    uart = UartController(port=PORT, baudrate=BAUD, dtr=False, rts=False, open_delay=0.5)
    if uart.serial is None:
        print("错误: 无法打开串口")
        return 1

    try:
        # 待机
        uart.send_raw(build_cmd(0, 0, 0, 0))
        time.sleep(0.3)
        print("待机 OK")

        # 校准
        print("\n校准中 (2 秒)...")
        uart.send_raw(build_cmd(4, 0, 5, 0))
        time.sleep(2.0)
        uart.send_raw(build_cmd(0, 0, 0, 0))
        time.sleep(0.3)
        print("校准完成")

        # 45°
        print("\n[发送] 位置 45° (0.785 rad)")
        uart.send_raw(build_cmd(3, math.pi / 4, 3.0, 0.1))
        time.sleep(0.5)

        fb = read_feedback(uart, timeout=2.0)
        if fb:
            pos_deg = fb[0] * 57.2958
            print(f"[回传] 位置={pos_deg:.1f}°, 速度={fb[1]:.2f} rad/s, Vq={fb[2]:.3f} V")
        else:
            print("[回传] 超时，未收到回传帧")

        time.sleep(1.0)

        # 0°
        print("\n[发送] 位置 0°")
        uart.send_raw(build_cmd(3, 0, 3.0, 0.1))
        time.sleep(0.5)

        fb = read_feedback(uart, timeout=2.0)
        if fb:
            pos_deg = fb[0] * 57.2958
            print(f"[回传] 位置={pos_deg:.1f}°, 速度={fb[1]:.2f} rad/s, Vq={fb[2]:.3f} V")
        else:
            print("[回传] 超时，未收到回传帧")

        time.sleep(1.0)

        # 待机
        print("\n待机")
        uart.send_raw(build_cmd(0, 0, 0, 0))
        time.sleep(0.3)
        print("完成")

    except KeyboardInterrupt:
        print("\n中断")
        uart.send_raw(build_cmd(0, 0, 0, 0))
    finally:
        uart.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
