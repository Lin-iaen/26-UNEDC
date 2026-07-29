# USB 视频采集卡 Linux 使用教程

在任意 Linux 设备上使用 USB 视频采集卡（UVC 协议）接收 HDMI 信号并录制。

## 适用场景

- HDMI 源（电脑 / 游戏机 / 相机）→ HDMI 线 → USB 采集卡 → Linux 主机
- 单帧截图、视频录制、推流
- **不需要采集卡厂商的专用驱动或软件**

## 硬件要求

- Linux 主机（x86_64 / ARM64 均可）
- USB 视频采集卡（UVC 协议，如 Macrosilicon MS2109 / MS2130）
- HDMI 线 + 视频源

## 依赖安装

```bash
# Debian/Ubuntu/Raspberry Pi OS
sudo apt update
sudo apt install ffmpeg v4l-utils

# 如果编译需要（可选，用于 drm_fb_test 等高级功能）
sudo apt install libdrm-dev gcc
```

## 快速上手

### 1. 查找设备

```bash
v4l2-ctl --list-devices
```

输出示例：
```
UVC Camera (345f:2109): USB Vid (usb-xhci-hcd.0-1):
    /dev/video0
    /dev/video1
```

**记住设备节点**（通常是 `/dev/video0`）。

### 2. 查看支持的格式与分辨率

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

常见输出：
```
[0]: 'MJPG' (Motion-JPEG, compressed)
     Size: Discrete 1920x1080 @ 60/50/30/20/10 fps
     Size: Discrete 1280x720 @ 60/50/30/20/10 fps
     Size: Discrete 640x480  @ 60/50/30/20/10 fps
[1]: 'YUYV' (YUYV 4:2:2)
     Size: Discrete 1920x1080 @ 10/5 fps
     Size: Discrete 640x480  @ 60/30/20/10 fps
```

**MJPEG 模式**推荐：硬件压缩，CPU 占用低。

### 3. 采集一帧

```bash
# MJPEG 模式（推荐）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 \
       -vframes 1 frame.jpg

# YUYV 原始格式
ffmpeg -f v4l2 -input_format yuyv422 -video_size 640x480 -i /dev/video0 \
       -vframes 1 frame.png
```

### 4. 录制视频

```bash
# 录制 10 秒，直接存 MJPEG 流（不转码，CPU 几乎无负载）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 \
       -t 10 -c copy recording.mkv

# 录制并转码为 H.264（文件更小，需要 CPU）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 \
       -t 10 -c libx264 -preset ultrafast -crf 23 recording.mp4

# 录制并转码为 H.265（更小文件，较新 CPU 支持）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 \
       -t 10 -c libx265 -preset ultrafast -crf 28 recording.mp4
```

### 5. 实时预览

```bash
# 方式一：ffplay（ffmpeg 自带）
ffplay -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0

# 方式二：MJPEG HTTP 推流（浏览器观看）
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 \
       -f mjpeg -q 5 http://localhost:8080

# 方式三：保存为图片序列
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 \
       -vf fps=1 frames/img_%04d.jpg
```

## 常用参数说明

| 参数 | 说明 |
|---|---|
| `-f v4l2` | 指定输入格式为 V4L2 |
| `-input_format mjpeg` | 使用 MJPEG 压缩格式（硬件解码） |
| `-input_format yuyv422` | 使用 YUYV 原始格式 |
| `-video_size 1920x1080` | 采集分辨率 |
| `-framerate 30` | 采集帧率（YUYV 模式生效） |
| `-t 10` | 录制时长（秒） |
| `-c copy` | 直接复制流（不重新编码） |
| `-c libx264` | 用 H.264 编码 |
| `-preset ultrafast` | 编码速度最快（牺牲压缩率） |
| `-crf 23` | 画质（0-51，越小越好，23 为默认） |
| `-vframes 1` | 只取一帧 |

## 常见问题

### Q: 设备节点找不到（`/dev/video0` 不存在）

```bash
# 检查 USB 是否识别
lsusb | grep -iE "video|UVC|345f"

# 检查内核模块
lsmod | grep uvcvideo

# 如果识别到但没有 video 节点，重新插拔 USB
```

### Q: 彩色竖条纹

**原因**：读到了错误的像素格式。

**解决**：指定正确的 `-input_format`。采集卡通常是 MJPEG 或 YUYV：

```bash
# 先列出支持的格式
v4l2-ctl -d /dev/video0 --list-formats

# 然后指定对应的格式
ffmpeg -f v4l2 -input_format mjpeg -i /dev/video0 ...   # 或
ffmpeg -f v4l2 -input_format yuyv422 -i /dev/video0 ...
```

### Q: `Device or resource busy`

设备被其他进程占用：

```bash
# 查找占用进程
lsof /dev/video0
# 或
fuser /dev/video0

# 杀死占用进程
sudo fuser -k /dev/video0
```

### Q: 权限不足

```bash
# 将用户加入 video 组
sudo usermod -aG video $USER
# 重新登录生效
```

### Q: USB 反复断开（`error -110`）

供电不足。解决方案：
1. 使用 **外接供电的 USB Hub**
2. 换一根短的高质量 USB 线
3. 如果连接 HDMI 后出现，尝试屏蔽 HDMI 热插拔检测或用独立 HDMI 源

### Q: 画面延迟大

采集卡本身有 1-2 帧延迟（约 30-60ms），这是硬件特性。降低分辨率可以减少延迟。

## 进阶：树莓派 5 自环采集

在 Pi 5 上实现「HDMI 输出 → 采集卡 → USB 回自己」需要额外的步骤：

1. **UVC 流唤醒 EDID**：采集卡在推流后才暴露 EDID
2. **DRM 模式设置**：用 `drmModeSetCrtc` 让 HDMI 输出有效画面
3. **帧缓冲持续更新**：摄像头画面写入 dumb buffer

参见本仓库的 `tests/drm_fb_test.c` 和 `tests/record_hdmi_loopback.py`。

```bash
# 自环录制（需要外接供电 USB Hub 稳定连接）
python tests/record_hdmi_loopback.py --duration 10 --size 640x480
```

## 配套文件

| 文件 | 说明 |
|---|---|
| `tests/test_uvc_capture.py` | UVC 采集卡自动测试脚本 |
| `tests/drm_fb_test.c` | DRM dumb buffer + drmModeSetCrtc（Pi 5自环用） |
| `tests/record_hdmi_loopback.py` | 摄像头→HDMI→采集卡→录制（Pi 5自环用） |
