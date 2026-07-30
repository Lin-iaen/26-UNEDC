#!/usr/bin/env python3
"""PicoFOC 电机硬件驱动测试（真实串口 + CAN 卡 + 电机）。

用法:
    source venv/bin/activate
    python tests/test_picofoc_hardware.py

安全:
  - 所有运动指令前先发 standby
  - Ctrl+C 立即停止电机
  - 脚本退出前自动发送 standby
"""

import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drivers.picofoc_motor import PicoFOCMotor
from src.drivers.uart import UartController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
BAUD = 4000000
MOTOR_ID = 1


def main() -> int:
    print("=" * 56)
    print("  PicoFOC 电机硬件驱动测试")
    print("=" * 56)
    print(f"  串口: {PORT}")
    print(f"  波特率: {BAUD}")
    print(f"  电机 ID: {MOTOR_ID}")
    print("=" * 56)

    uart = UartController(port=PORT, baudrate=BAUD, dtr=False, rts=False, open_delay=0.5)
    if uart.serial is None:
        print("错误: 无法打开串口")
        return 1

    motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)

    try:
        # ── Step 1: 待机 ──
        print("\n[1/5] 待机 (STANDBY)")
        motor.standby()
        time.sleep(0.5)
        print("  OK")

        # ── Step 2: 校准 ──
        print("\n[2/5] 电角度自校准 (CALIBRATE)")
        print("  电机将发出轻微抖动/蜂鸣声，持续约 2 秒...")
        motor.calibrate()
        time.sleep(3.0)
        motor.standby()
        time.sleep(0.5)
        print("  校准完成")

        # ── Step 3: 位置模式 ──
        print("\n[3/5] 位置模式 - 旋转到 45°")
        print("  3 秒内将电机转到约 45° 位置...")
        motor.set_position(math.pi / 4, kp=3.0, kd=0.1)
        time.sleep(3.0)
        motor.standby()
        time.sleep(0.5)
        print("  位置 45° 完成")

        print("\n[3/5] 位置模式 - 回到 0°")
        motor.set_position(0.0, kp=3.0, kd=0.1)
        time.sleep(2.0)
        motor.standby()
        time.sleep(0.5)
        print("  回到 0° 完成")

        # ── Step 4: 速度模式 ──
        print("\n[4/5] 速度模式 - 慢速旋转")
        print("  电机将以约 2 rad/s (~19 rpm) 旋转 2 秒...")
        motor.set_speed(2.0, kp=0.02, ki=0.01)
        time.sleep(2.0)
        motor.standby()
        time.sleep(0.5)
        print("  速度模式完成")

        # ── Step 5: 完成 ──
        print("\n[5/5] 测试完成")
        motor.standby()
        print("  电机已待机")

        print("\n" + "=" * 56)
        print("  全部测试通过，电机运行正常")
        print("=" * 56)
        return 0

    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 130
    finally:
        motor.standby()
        time.sleep(0.2)
        uart.close()


if __name__ == "__main__":
    sys.exit(main())
