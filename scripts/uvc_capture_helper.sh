#!/usr/bin/env bash
# USB 采集卡快速使用助手
# 用法: bash scripts/uvc_capture_helper.sh [--record DURATION] [--snapshot]

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# 查找采集卡设备
find_uvc() {
    local dev
    for dev in /dev/video*; do
        if v4l2-ctl -d "$dev" --info 2>/dev/null | grep -q "UVC Camera"; then
            echo "$dev"
            return 0
        fi
    done
    return 1
}

check_deps() {
    local missing=0
    for cmd in ffmpeg v4l2-ctl; do
        if ! command -v "$cmd" &>/dev/null; then
            error "缺少依赖: $cmd"
            info "安装: sudo apt install ffmpeg v4l-utils"
            missing=1
        fi
    done
    return $missing
}

show_info() {
    local dev="$1"
    echo "=========================================="
    echo "  采集卡设备: $dev"
    echo "=========================================="
    v4l2-ctl -d "$dev" --all | head -20
    echo ""
    echo "支持的分辨率/格式:"
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
        file "$out"
    else
        # 重试 YUYV
        out2="${out%.*}.png"
        ffmpeg -hide_banner -f v4l2 -input_format yuyv422 \
               -video_size 640x480 -i "$dev" \
               -vframes 1 -update 1 -y "$out2" 2>/dev/null
        if [ -f "$out2" ]; then
            info "MJPEG 失败，YUYV 成功: $out2"
        else
            error "采集失败"
        fi
    fi
}

record() {
    local dev="$1" dur="$2" out="${3:-recording.mp4}"
    info "录制 ${dur}秒 → $out"
    ffmpeg -hide_banner -stats \
           -f v4l2 -input_format mjpeg \
           -video_size 1920x1080 -i "$dev" \
           -t "$dur" \
           -c libx264 -preset ultrafast -crf 23 \
           -pix_fmt yuv420p \
           -y "$out"
    info "完成: $out ($(du -h "$out" | cut -f1))"
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
    echo ""
    echo "请检查:"
    echo "  1. lsusb | grep -iE 'video|UVC|345f'  — USB 是否识别"
    echo "  2. sudo modprobe uvcvideo              — 内核模块"
    echo "  3. 重新插拔 USB"
    exit 1
fi

info "检测到采集卡: $DEV"
echo ""

# 无参数：显示信息
if [ $# -eq 0 ]; then
    show_info "$DEV"
    echo ""
    echo "可用命令:"
    echo "  $0 --snapshot [filename]   采集一帧"
    echo "  $0 --record <秒> [filename] 录制视频"
    echo "  $0 --info                   显示设备详情"
    exit 0
fi

case "${1:-}" in
    --info|-i)
        show_info "$DEV"
        ;;
    --snapshot|-s)
        snapshot "$DEV" "${2:-capture.jpg}"
        ;;
    --record|-r)
        dur="${2:-10}"
        record "$DEV" "$dur" "${3:-recording.mp4}"
        ;;
    *)
        error "未知参数: $1"
        echo "用法: $0 [--snapshot|--record <秒>|--info]"
        exit 1
        ;;
esac
