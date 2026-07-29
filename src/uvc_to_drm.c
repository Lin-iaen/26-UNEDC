/*
 * uvc_to_drm — USB 采集卡画面直接输出到 HDMI 显示屏
 *
 * 从 V4L2 设备读 YUYV 帧，转换为 RGB，通过 DRM/KMS 输出到 HDMI。
 * 不需要桌面环境、X11、Wayland。
 *
 * 编译:
 *   gcc -o uvc_to_drm src/uvc_to_drm.c -ldrm \
 *       -I/usr/include/libdrm -I/usr/include/drm
 *
 * 用法:
 *   sudo ./uvc_to_drm /dev/video0 /dev/dri/card1 [WxH]
 *
 *   默认 640x480，支持 1920x1080、1280x720 等。
 *   按 Ctrl+C 退出。
 *
 * 依赖:
 *   sudo apt install libdrm-dev gcc
 *   采集卡必须支持 YUYV 格式（所有 UVC 采集卡都支持）
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))

static volatile int keep_running = 1;
static int drm_fd = -1;

static void handle_signal(int sig) {
    (void)sig;
    keep_running = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * V4L2 — 打开采集卡并设置 YUYV 格式
 * ═══════════════════════════════════════════════════════════════════════════ */

static int v4l2_open(const char *dev, int w, int h) {
    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "无法打开 %s: %s\n", dev, strerror(errno));
        return -1;
    }

    struct v4l2_format vfmt = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .fmt.pix = {
            .width = w, .height = h,
            .pixelformat = V4L2_PIX_FMT_YUYV,
            .field = V4L2_FIELD_NONE,
        },
    };
    if (ioctl(fd, VIDIOC_S_FMT, &vfmt) < 0) {
        fprintf(stderr, "VIDIOC_S_FMT(YUYV) 失败: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    printf("V4L2: %s  YUYV %dx%d\n", dev, vfmt.fmt.pix.width, vfmt.fmt.pix.height);
    return fd;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DRM — 查找显示器、创建 framebuffer、设置显示模式
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t fb_id, crtc_id, conn_id;
    uint8_t *map;
    size_t pitch, size;
} DrmFb;

static drmModeConnector *drm_find_connector(int fd) {
    drmModeRes *res = drmModeGetResources(fd);
    if (!res) return NULL;
    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors; i++) {
        conn = drmModeGetConnector(fd, res->connectors[i]);
        if (conn && conn->connection == DRM_MODE_CONNECTED) break;
        if (conn) { drmModeFreeConnector(conn); conn = NULL; }
    }
    if (!conn && res->count_connectors > 0)
        conn = drmModeGetConnector(fd, res->connectors[0]);
    drmModeFreeResources(res);
    return conn;
}

static drmModeModeInfo *drm_preferred_mode(drmModeConnector *conn) {
    for (int i = 0; i < conn->count_modes; i++)
        if (conn->modes[i].type & DRM_MODE_TYPE_PREFERRED)
            return &conn->modes[i];
    return conn->count_modes > 0 ? &conn->modes[0] : NULL;
}

static uint32_t drm_find_crtc(int fd, drmModeConnector *conn) {
    drmModeRes *res = drmModeGetResources(fd);
    if (!res) return 0;
    uint32_t id = 0;
    for (int i = 0; i < res->count_encoders; i++) {
        drmModeEncoder *enc = drmModeGetEncoder(fd, res->encoders[i]);
        if (!enc) continue;
        if (enc->encoder_id == conn->encoder_id && enc->crtc_id) { id = enc->crtc_id; drmModeFreeEncoder(enc); break; }
        drmModeFreeEncoder(enc);
    }
    if (!id) {
        for (int j = 0; j < conn->count_encoders && !id; j++) {
            for (int i = 0; i < res->count_encoders; i++) {
                drmModeEncoder *enc = drmModeGetEncoder(fd, res->encoders[i]);
                if (!enc || enc->encoder_id != conn->encoders[j]) { if (enc) drmModeFreeEncoder(enc); continue; }
                for (int k = 0; k < res->count_crtcs; k++) {
                    if (enc->possible_crtcs & (1 << k)) { id = res->crtcs[k]; drmModeFreeEncoder(enc); goto done; }
                }
                drmModeFreeEncoder(enc);
            }
        }
    }
done:
    drmModeFreeResources(res);
    return id;
}

static int drm_setup(DrmFb *fb, int w, int h) {
    struct drm_mode_create_dumb create = {.width = w, .height = h, .bpp = 32};
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) < 0) return -1;
    uint32_t handles[4] = {create.handle}, pitches[4] = {create.pitch}, offsets[4] = {0};
    if (drmModeAddFB2(drm_fd, w, h, DRM_FORMAT_XRGB8888, handles, pitches, offsets, &fb->fb_id, 0) < 0) return -1;
    struct drm_mode_map_dumb map = {.handle = create.handle};
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map) < 0) return -1;
    fb->map = mmap(NULL, create.size, PROT_WRITE, MAP_SHARED, drm_fd, map.offset);
    if (fb->map == MAP_FAILED) return -1;
    fb->pitch = create.pitch;
    fb->size = create.size;
    printf("DRM: fb_id=%u %dx%d pitch=%u\n", fb->fb_id, w, h, create.pitch);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * YUYV → XRGB8888 转换
 * ═══════════════════════════════════════════════════════════════════════════ */

static void yuyv_to_xrgb(const uint8_t *src, uint32_t *dst,
                          int w, int h, int stride) {
    for (int y = 0; y < h; y++) {
        const uint8_t *line = src + y * stride;
        uint32_t *row = dst + y * w;
        for (int x = 0; x < w; x += 2) {
            int Y0 = line[0] - 16, U = line[1] - 128;
            int Y1 = line[2] - 16, V = line[3] - 128;
            line += 4;

            row[0] = (0xFF << 24)
                   | (CLAMP((298*Y0 + 409*V + 128) >> 8, 0, 255) << 16)
                   | (CLAMP((298*Y0 - 100*U - 208*V + 128) >> 8, 0, 255) << 8)
                   | CLAMP((298*Y0 + 516*U + 128) >> 8, 0, 255);
            row[1] = (0xFF << 24)
                   | (CLAMP((298*Y1 + 409*V + 128) >> 8, 0, 255) << 16)
                   | (CLAMP((298*Y1 - 100*U - 208*V + 128) >> 8, 0, 255) << 8)
                   | CLAMP((298*Y1 + 516*U + 128) >> 8, 0, 255);
            row += 2;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <V4L2_DEV> <DRM_DEV> [WxH]\n", argv[0]);
        fprintf(stderr, "示例: %s /dev/video0 /dev/dri/card1 640x480\n", argv[0]);
        fprintf(stderr, "      %s /dev/video0 /dev/dri/card1 1920x1080\n", argv[0]);
        return 1;
    }

    const char *v4l2_dev = argv[1], *drm_dev = argv[2];
    int w = 640, h = 480;
    if (argc > 3 && sscanf(argv[3], "%dx%d", &w, &h) != 2) {
        fprintf(stderr, "格式错误: %s (应为 WxH, 如 640x480)\n", argv[3]);
        return 1;
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* ── 打开 V4L2 ── */
    int v4l2_fd = v4l2_open(v4l2_dev, w, h);
    if (v4l2_fd < 0) return 1;

    /* ── 打开 DRM ── */
    drm_fd = open(drm_dev, O_RDWR);
    if (drm_fd < 0) { fprintf(stderr, "无法打开 %s: %s\n", drm_dev, strerror(errno)); close(v4l2_fd); return 1; }
    drmSetMaster(drm_fd);

    drmModeConnector *conn = drm_find_connector(drm_fd);
    if (!conn) { fprintf(stderr, "无可用 HDMI 显示器\n"); close(v4l2_fd); close(drm_fd); return 1; }
    printf("HDMI: connector %d (%s)\n", conn->connector_id,
           conn->connection == DRM_MODE_CONNECTED ? "connected" : "disconnected");

    drmModeModeInfo *mode = drm_preferred_mode(conn);
    if (!mode) { fprintf(stderr, "无可用显示模式\n"); goto cleanup; }

    DrmFb fb = {0};
    fb.crtc_id = drm_find_crtc(drm_fd, conn);
    fb.conn_id = conn->connector_id;
    if (!fb.crtc_id) { fprintf(stderr, "无可用 CRTC\n"); goto cleanup; }
    if (drm_setup(&fb, w, h) < 0) { fprintf(stderr, "framebuffer 创建失败\n"); goto cleanup; }
    if (drmModeSetCrtc(drm_fd, fb.crtc_id, fb.fb_id, 0, 0, &fb.conn_id, 1, mode) < 0) {
        fprintf(stderr, "drmModeSetCrtc 失败: %s\n", strerror(errno)); goto cleanup;
    }
    printf("显示已激活 — 按 Ctrl+C 退出\n");

    /* ── 主循环：读采集卡 → 显示 ── */
    int yuyv_stride = w * 2;
    size_t yuyv_size = (size_t)h * yuyv_stride;
    uint8_t *yuyv_buf = malloc(yuyv_size);
    if (!yuyv_buf) { fprintf(stderr, "malloc 失败\n"); goto cleanup; }

    while (keep_running) {
        ssize_t n = read(v4l2_fd, yuyv_buf, yuyv_size);
        if (n < 0) { if (errno == EINTR) continue; break; }
        if ((size_t)n < yuyv_size) continue;

        yuyv_to_xrgb(yuyv_buf, (uint32_t *)fb.map, w, h, yuyv_stride);
        drmModeDirtyFB(drm_fd, fb.fb_id, NULL, 0);
    }

    free(yuyv_buf);

cleanup:
    if (fb.map) munmap(fb.map, fb.size);
    if (fb.fb_id) drmModeRmFB(drm_fd, fb.fb_id);
    if (conn) drmModeFreeConnector(conn);
    close(v4l2_fd);
    if (drm_fd >= 0) close(drm_fd);
    printf("退出\n");
    return 0;
}
