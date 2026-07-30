#!/usr/bin/env python3
"""PicoFOC 电机 Web 控制面板。

用法:
    source venv/bin/activate
    python tests/test_picofoc_webcontrol.py
    浏览器打开 http://<树莓派IP>:5000

控制:
  - 模式选择 (待机/速度/位置/校准)
  - 目标值滑块 (rad 或 rad/s)
  - Kp / KdKi 增益调节
  - 实时状态显示
"""

import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, render_template_string

from src.drivers.picofoc_motor import PicoFOCMotor
from src.drivers.uart import UartController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("werkzeug").setLevel(logging.ERROR)

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
BAUD = 4000000
MOTOR_ID = 1
HTTP_PORT = 5000

app = Flask(__name__)

_motor: PicoFOCMotor | None = None
_motor_state: dict = {
    "mode": "STANDBY",
    "target": 0.0,
    "kp": 0.0,
    "kdki": 0.0,
    "connected": False,
    "feedback": None,
}
_lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PicoFOC 电机控制</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
  h1 { color: #e94560; margin-bottom: 20px; font-size: 1.4rem; }
  .card { background: #16213e; border-radius: 10px; padding: 20px; margin-bottom: 16px; }
  .card h2 { font-size: 1rem; color: #0f3460; margin-bottom: 12px; }
  label { display: block; margin: 10px 0 4px; font-size: 0.85rem; color: #a8a8b3; }
  select, input[type=range] { width: 100%; margin-bottom: 8px; }
  select { background: #0f3460; color: #eee; border: 1px solid #533483; padding: 8px; border-radius: 6px; font-size: 1rem; }
  input[type=range] { -webkit-appearance: none; height: 6px; border-radius: 3px; background: #533483; outline: none; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%; background: #e94560; cursor: pointer; }
  .val { float: right; color: #e94560; font-weight: bold; font-size: 0.9rem; }
  .btn { width: 100%; padding: 14px; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: 0.2s; }
  .btn-send { background: #e94560; color: #fff; }
  .btn-send:hover { background: #c73650; }
  .btn-send:active { transform: scale(0.97); }
  .btn-standby { background: #533483; color: #fff; margin-top: 10px; }
  .btn-standby:hover { background: #442a6e; }
  #status { margin-top: 16px; padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.9rem; line-height: 1.6; }
  .ok { background: #1a3a2a; color: #4ade80; }
  .err { background: #3a1a1a; color: #f87171; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }
  @media (max-width: 480px) { .grid { grid-template-columns: 1fr; } }
  .fb { font-family: monospace; font-size: 0.9rem; line-height: 1.8; }
  .fb td { padding: 2px 12px 2px 0; }
  .fb .val { color: #e94560; font-weight: bold; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-left: 6px; }
  .badge-green { background: #166534; color: #86efac; }
  .badge-red { background: #7f1d1d; color: #fca5a5; }
</style>
</head>
<body>
<h1>⚙ PicoFOC 电机控制</h1>

<div class="card">
  <h2>运行模式</h2>
  <select id="mode">
    <option value="STANDBY">待机 (STANDBY)</option>
    <option value="POSITION">位置 (POSITION)</option>
    <option value="SPEED">速度 (SPEED)</option>
    <option value="CALIBRATE">校准 (CALIBRATE)</option>
  </select>
</div>

<div class="card">
  <h2>参数</h2>
  <div class="grid">
    <div>
      <label>目标值 <span id="targetVal" class="val">0.00</span></label>
      <input type="range" id="target" min="-6.28" max="6.28" step="0.01" value="0">
    </div>
    <div>
      <label>Kp <span id="kpVal" class="val">5.00</span></label>
      <input type="range" id="kp" min="0" max="10" step="0.01" value="5">
    </div>
  </div>
  <div class="grid">
    <div>
      <label>Kd <span id="kdVal" class="val">0.00</span></label>
      <input type="range" id="kd" min="0" max="1" step="0.01" value="0">
    </div>
    <div></div>
  </div>
</div>

<button class="btn btn-send" onclick="sendCmd()">发送指令</button>
<button class="btn btn-standby" onclick="sendStandby()">紧急待机 (STOP)</button>

<div class="card">
  <h2>回传反馈</h2>
  <table class="fb">
    <tr><td>位置</td><td class="val" id="fbPos">--</td><td>deg</td></tr>
    <tr><td>速度</td><td class="val" id="fbSpd">--</td><td>rad/s</td></tr>
    <tr><td>Vq</td><td class="val" id="fbVq">--</td><td>V</td></tr>
  </table>
</div>

<div id="status" class="ok">就绪</div>

<script>
let statusOk = true;

document.getElementById('target').oninput = function() {
  document.getElementById('targetVal').textContent = this.value;
};
document.getElementById('kp').oninput = function() {
  document.getElementById('kpVal').textContent = parseFloat(this.value).toFixed(2);
};
document.getElementById('kd').oninput = function() {
  document.getElementById('kdVal').textContent = parseFloat(this.value).toFixed(2);
};

function sendCmd() {
  const mode = document.getElementById('mode').value;
  const target = parseFloat(document.getElementById('target').value);
  const kp = parseFloat(document.getElementById('kp').value);
  const kd = parseFloat(document.getElementById('kd').value);
  fetch('/cmd', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode, target, kp, kd})
  }).then(r => r.json()).then(d => {
    setStatus(d.status, d.ok);
  }).catch(e => setStatus('请求失败: ' + e, false));
}

function sendStandby() {
  document.getElementById('mode').value = 'STANDBY';
  fetch('/cmd', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode:'STANDBY', target:0, kp:0, kd:0})
  }).then(r => r.json()).then(d => {
    setStatus(d.status, d.ok);
  }).catch(e => setStatus('请求失败: ' + e, false));
}

function setStatus(msg, ok) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = ok ? 'ok' : 'err';
}

function fetchStatus() {
  fetch('/status').then(r => r.json()).then(d => {
    const el = document.getElementById('status');
    el.textContent = '电机 ' + (d.connected ? '已连接' : '未连接')
      + ' | 模式: ' + d.mode
      + ' | target: ' + d.target.toFixed(2)
      + ' | Kp: ' + d.kp.toFixed(3)
      + ' | Kd/Ki: ' + d.kdki.toFixed(3);
    el.className = d.connected ? 'ok' : 'err';

    const fb = d.feedback;
    if (fb) {
      document.getElementById('fbPos').textContent = (fb.pos_deg || 0).toFixed(1);
      document.getElementById('fbSpd').textContent = (fb.spd || 0).toFixed(2);
      document.getElementById('fbVq').textContent = (fb.vq || 0).toFixed(3);
    }
  }).catch(() => {});
}
setInterval(fetchStatus, 2000);
fetchStatus();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/status")
def status():
    with _lock:
        state = dict(_motor_state)
        if state.get("feedback"):
            state["feedback"]["pos_deg"] = state["feedback"].get("pos_deg", 0.0)
        return jsonify(**state)


@app.route("/cmd", methods=["POST"])
def cmd():
    global _motor
    data = request.get_json(force=True)
    mode = data.get("mode", "STANDBY")
    target = float(data.get("target", 0.0))
    kp = float(data.get("kp", 0.0))
    kd = float(data.get("kd", 0.0))

    with _lock:
        _motor_state["target"] = target
        _motor_state["kp"] = kp
        _motor_state["kdki"] = kd
        _motor_state["mode"] = mode

    if _motor is None:
        return jsonify(ok=False, status="电机未初始化")

    try:
        if mode == "STANDBY":
            _motor.standby()
        elif mode == "SPEED":
            _motor.set_speed(target, kp=kp, ki=kd)
        elif mode == "POSITION":
            _motor.set_position(target, kp=kp, kd=kd)
        elif mode == "CALIBRATE":
            _motor.calibrate()
        else:
            return jsonify(ok=False, status=f"未知模式: {mode}")

        with _lock:
            _motor_state["connected"] = True

        return jsonify(ok=True, status=f"{mode} → target={target:.3f}, Kp={kp:.3f}, Kd/Ki={kd:.3f}")
    except Exception as e:
        logger.exception("指令执行失败")
        return jsonify(ok=False, status=f"错误: {e}")


def main():
    global _motor
    print("=" * 56)
    print("  PicoFOC 电机 Web 控制面板")
    print("=" * 56)
    print(f"  串口: {PORT} @ {BAUD}")
    print(f"  电机 ID: {MOTOR_ID}")
    print(f"  网页: http://<树莓派IP>:{HTTP_PORT}")
    print("=" * 56)
    print()

    uart = UartController(port=PORT, baudrate=BAUD, dtr=False, rts=False, open_delay=0.5)
    if uart.serial is None:
        print("错误: 无法打开串口")
        sys.exit(1)

    _motor = PicoFOCMotor(uart, motor_id=MOTOR_ID)
    _motor.standby()

    with _lock:
        _motor_state["connected"] = True

    def _feedback_loop():
        while True:
            if _motor is None:
                break
            fb = _motor.read_feedback()
            if fb is not None:
                with _lock:
                    _motor_state["feedback"] = {
                        "pos_deg": fb.position_rad * 57.2958,
                        "spd": fb.speed_rads,
                        "vq": fb.vq,
                    }
            time.sleep(0.1)

    threading.Thread(target=_feedback_loop, daemon=True).start()

    try:
        app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if _motor is not None:
            _motor.standby()
        uart.close()
        print("电机已待机，串口已关闭")


if __name__ == "__main__":
    main()
