#!/usr/bin/env bash
# USB 采集卡快速使用助手
# 用法: bash scripts/uvc_capture_helper.sh [command]

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

find_uvc() {
    for dev in /dev/video*; do
        if v4l2-ctl -d "$dev" --info 2>/dev/null | grep -q "UVC Camera"; then
            echo "$dev"; return 0
        fi
    done
    return 1
}

check_deps() {
    local missing=0
    for cmd in ffmpeg v4l2-ctl gcc; do
        if ! command -v "$cmd" &>/dev/null 2>&1; then
            error "缺少依赖: $cmd (sudo apt install ffmpeg v4l-utils gcc)"
            missing=1
        fi
    done
    return $missing
}

show_info() {
    local dev="$1"
    echo "=========================================="
    echo "  采集卡: $dev"
    echo "=========================================="
    v4l2-ctl -d "$dev" --all | head -20
    echo ""; echo "支持的分辨率/格式:"
    v4l2-ctl -d "$dev" --list-formats-ext
    echo "=========================================="
}

snapshot() {
    local dev="$1" out="${2:-capture.jpg}"
    info "采集一帧 → $out"
    ffmpeg -hide_banner -f v4l2 -input_format mjpeg \
           -video_size 1920x1080 -i "$dev" \
           -vframes 1 -update 1 -y "$out" 2>/dev/null
    if [ -f "$out" ]; then
        info "完成: $out ($(du -h "$out" | cut -f1))"
    else
        out2="${out%.*}.png"
        ffmpeg -hide_banner -f v4l2 -input_format yuyv422 \
               -video_size 640x480 -i "$dev" \
               -vframes 1 -update 1 -y "$out2" 2>/dev/null
        if [ -f "$out2" ]; then
            info "MJPEG 失败，YUYV 成功: $out2"
        else
            error "采集失败，尝试: v4l2-ctl -d $dev --list-formats"
        fi
    fi
}

record() {
    local dev="$1" dur="$2" out="${3:-recording.mp4}"
    info "录制 ${dur}秒 → $out"
    ffmpeg -hide_banner -stats \
           -f v4l2 -input_format mjpeg -video_size 1920x1080 -i "$dev" \
           -t "$dur" -c libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p -y "$out"
    info "完成: $out ($(du -h "$out" | cut -f1))"
}

display() {
    local dev="$1" size="${2:-640x480}"
    info "实时显示: $dev → DRM HDMI (${size})"
    info "按 Ctrl+C 退出"

    # 查找 DRM 设备
    local drm_dev=""
    for d in /dev/dri/card*; do
        if [ -e "$d" ]; then drm_dev="$d"; break; fi
    done
    if [ -z "$drm_dev" ]; then
        error "未找到 DRM 设备 (/dev/dri/card*)"
        error "需要: sudo apt install libdrm-dev"
        exit 1
    fi

    # 尝试编译 uvc_to_drm
    local UVC2DRM=""
    if command -v gcc &>/dev/null; then
        UVC2DRM=$(mktemp /tmp/uvc2drm_XXXXXX)
        gcc -o "$UVC2DRM" "$(dirname "$0")/../src/uvc_to_drm.c" \
            -ldrm -I/usr/include/libdrm -I/usr/include/drm 2>/dev/null || UVC2DRM=""
    fi

    if [ -n "$UVC2DRM" ] && [ -x "$UVC2DRM" ]; then
        sudo "$UVC2DRM" "$dev" "$drm_dev" "$size"
        rm -f "$UVC2DRM"
    elif command -v gst-launch-1.0 &>/dev/null; then
        info "使用 GStreamer kmssink"
        gst-launch-1.0 v4l2src device="$dev" ! \
            image/jpeg,width="${size%x*}",height="${size#*x}" ! \
            jpegdec ! videoconvert ! kmssink driver-name=vc4
    else
        error "需要编译 uvc_to_drm 或安装 GStreamer"
        error "  sudo apt install libdrm-dev gcc"
        error "  或: sudo apt install gstreamer1.0-..."
        exit 1
    fi
}

list_devices() {
    echo "系统中所有 V4L2 设备:"
    v4l2-ctl --list-devices
}

# ── 主流程 ──

check_deps || exit 1
DEV=$(find_uvc)

if [ -z "$DEV" ]; then
    error "未检测到 UVC 采集卡"
    list_devices
    echo ""; echo "请检查:"
    echo "  1. lsusb | grep -iE 'video|UVC|345f'  — USB 是否识别"
    echo "  2. sudo modprobe uvcvideo              — 内核模块"
    echo "  3. 重新插拔 USB"
    exit 1
fi

info "检测到采集卡: $DEV"

case "${1:-}" in
    --info|-i)  show_info "$DEV" ;;
    --snapshot|-s) snapshot "$DEV" "${2:-capture.jpg}" ;;
    --record|-r)  record "$DEV" "${2:-10}" "${3:-recording.mp4}" ;;
    --display|-d) display "$DEV" "${2:-640x480}" ;;
    *)
        echo "用法: $0 [命令]"
        echo ""
        echo "命令:"
        echo "  --snapshot [文件名]    采集一帧"
        echo "  --record <秒> [文件名]  录制视频"
        echo "  --display [分辨率]     实时显示到 HDMI"
        echo "  --info                 显示设备详情"
        echo ""
        echo "示例:"
        echo "  $0 --snapshot frame.jpg"
        echo "  $0 --record 10 video.mp4"
        echo "  $0 --display 640x480"
        ;;
esac
