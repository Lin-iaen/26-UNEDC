/*
 * drm_fb_test — 用 libdrm 分配 dumb buffer → 彩条图案 → drmModeSetCrtc
 *
 * 编译:
 *   gcc -o tests/drm_fb_test tests/drm_fb_test.c -ldrm
 *
 * 用法:
 *   # 先启动 UVC 流唤醒 HDMI，再运行本程序
 *   ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 \
 *          -i /dev/video8 -t 30 -f null - &
 *   sleep 4
 *   ./tests/drm_fb_test /dev/dri/card1
 *
 * 程序持续运行直到 Ctrl+C。此时用 ffmpeg 从 /dev/video8 采集即可。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/mman.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

static volatile int keep_running = 1;

static void handle_signal(int sig) {
    (void)sig;
    keep_running = 0;
}

/* 查找指定 connector_id 或第一个 connected 的 connector */
static drmModeConnector *get_connector(int fd, int want_id) {
    drmModeRes *res = drmModeGetResources(fd);
    if (!res) {
        fprintf(stderr, "drmModeGetResources failed\n");
        return NULL;
    }

    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors; i++) {
        if (want_id > 0 && res->connectors[i] != (uint32_t)want_id)
            continue;
        conn = drmModeGetConnector(fd, res->connectors[i]);
        if (conn && conn->connection == DRM_MODE_CONNECTED) {
            break;
        }
        if (conn) {
            drmModeFreeConnector(conn);
            conn = NULL;
        }
        if (want_id > 0)
            break;  /* 指定了 ID 但没 connected 也要退出 */
    }

    if (!conn) {
        /* 没找到 connected，回退到任意 connector */
        for (int i = 0; i < res->count_connectors; i++) {
            if (want_id > 0 && res->connectors[i] != (uint32_t)want_id)
                continue;
            conn = drmModeGetConnector(fd, res->connectors[i]);
            if (conn) break;
            if (want_id > 0) break;
        }
    }

    drmModeFreeResources(res);
    return conn;
}

/* 选 prefered mode，没有就取第一个 */
static drmModeModeInfo *get_preferred_mode(drmModeConnector *conn) {
    for (int i = 0; i < conn->count_modes; i++) {
        if (conn->modes[i].type & DRM_MODE_TYPE_PREFERRED)
            return &conn->modes[i];
    }
    if (conn->count_modes > 0)
        return &conn->modes[0];
    return NULL;
}

/* 找第一个可用的 encoder + CRTC */
static uint32_t find_crtc(int fd, drmModeConnector *conn) {
    drmModeRes *res = drmModeGetResources(fd);
    if (!res) return 0;

    uint32_t crtc_id = 0;

    /* 优先用 connector 当前绑定的 encoder 的 CRTC */
    for (int i = 0; i < res->count_encoders; i++) {
        drmModeEncoder *enc = drmModeGetEncoder(fd, res->encoders[i]);
        if (!enc) continue;
        if (enc->encoder_id == conn->encoder_id && enc->crtc_id) {
            crtc_id = enc->crtc_id;
            drmModeFreeEncoder(enc);
            goto done;
        }
        drmModeFreeEncoder(enc);
    }

    /* 否则从 connector 的可能的 encoder 中找 */
    for (int j = 0; j < conn->count_encoders; j++) {
        for (int i = 0; i < res->count_encoders; i++) {
            drmModeEncoder *enc = drmModeGetEncoder(fd, res->encoders[i]);
            if (!enc) continue;
            if (enc->encoder_id == conn->encoders[j]) {
                /* 找这个 encoder 能用的第一个 CRTC */
                for (int k = 0; k < res->count_crtcs; k++) {
                    if (enc->possible_crtcs & (1 << k)) {
                        crtc_id = res->crtcs[k];
                        drmModeFreeEncoder(enc);
                        goto done;
                    }
                }
                drmModeFreeEncoder(enc);
            }
        }
    }

done:
    drmModeFreeResources(res);
    return crtc_id;
}

/* 创建 dumb buffer 并 mmap */
static struct dumb_ctx {
    uint32_t fb_id;
    uint32_t handle;
    uint8_t  *map;
    size_t   pitch;
    size_t   size;
} create_dumb_fb(int fd, int width, int height) {
    struct dumb_ctx ctx = {0};

    struct drm_mode_create_dumb create = {
        .width = width,
        .height = height,
        .bpp = 32,
    };
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) < 0) {
        fprintf(stderr, "CREATE_DUMB failed: %s\n", strerror(errno));
        return ctx;
    }
    ctx.handle = create.handle;
    ctx.pitch = create.pitch;
    ctx.size = create.size;

    /* AddFB2 */
    uint32_t handles[4] = {ctx.handle};
    uint32_t pitches[4] = {ctx.pitch};
    uint32_t offsets[4] = {0};
    if (drmModeAddFB2(fd, width, height, DRM_FORMAT_XRGB8888,
                       handles, pitches, offsets, &ctx.fb_id, 0) < 0) {
        fprintf(stderr, "AddFB2 failed: %s\n", strerror(errno));
        return ctx;
    }

    /* mmap */
    struct drm_mode_map_dumb map = {.handle = ctx.handle};
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map) < 0) {
        fprintf(stderr, "MAP_DUMB failed: %s\n", strerror(errno));
        return ctx;
    }
    ctx.map = mmap(NULL, ctx.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd, map.offset);
    if (ctx.map == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        ctx.map = NULL;
    }

    return ctx;
}

/* 画 SMPTE 彩条 + 白框棋盘格 */
static void fill_test_pattern(uint8_t *buf, int width, int height, size_t pitch) {
    /* 7 色竖彩条 + 渐变 */
    static const uint32_t colors[8] = {
        0x00FFFFFF, /* 白 */
        0x0000FFFF, /* 黄 */
        0x00FFFF00, /* 青 */
        0x0000FF00, /* 绿 */
        0x00FF00FF, /* 品 */
        0x00FF0000, /* 红 */
        0x000000FF, /* 蓝 */
        0x00000000, /* 黑 */
    };

    for (int y = 0; y < height; y++) {
        uint32_t *row = (uint32_t *)(buf + y * pitch);
        for (int x = 0; x < width; x++) {
            int ci = x * 8 / width;
            uint32_t c = colors[ci];

            /* 棋盘格叠加 (80px 格子) */
            int cell_x = x / 80;
            int cell_y = y / 80;
            if ((cell_x + cell_y) % 2 == 0) {
                /* 每个格子加一点亮度渐变 */
                int bright = 255;
                row[x] = (0xFF << 24) | c;
            } else {
                /* 暗格 — 保留颜色但降低亮度 */
                int r = (c >> 16) & 0xFF;
                int g = (c >> 8) & 0xFF;
                int b = c & 0xFF;
                r = r * 3 / 10;
                g = g * 3 / 10;
                b = b * 3 / 10;
                row[x] = (0xFF << 24) | (r << 16) | (g << 8) | b;
            }

            /* 白框叠加 (边缘 4px) */
            if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) {
                row[x] = 0xFFFFFFFF;
            }
        }
    }

    /* 底部 1/4 加文字识别区域: 黑白相间横条 */
    for (int y = height * 3 / 4; y < height; y++) {
        uint32_t *row = (uint32_t *)(buf + y * pitch);
        int bar = (y / 20) % 2;
        uint32_t c = bar ? 0xFFFFFF : 0x000000;
        for (int x = width / 4; x < width * 3 / 4; x++) {
            row[x] = (0xFF << 24) | c;
        }
    }
}

int main(int argc, char **argv) {
    const char *dev = argc > 1 ? argv[1] : "/dev/dri/card1";
    int conn_id = argc > 2 ? atoi(argv[2]) : 0;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "cannot open %s: %s\n", dev, strerror(errno));
        return 1;
    }

    /* 检查 DRM master */
    if (drmSetMaster(fd) < 0) {
        fprintf(stderr, "drmSetMaster failed (already in use?): %s\n",
                strerror(errno));
        /* 仍然尝试继续 */
    }

    /* 找到 connector */
    drmModeConnector *conn = get_connector(fd, conn_id);
    if (!conn) {
        fprintf(stderr, "no suitable connector found\n");
        close(fd);
        return 1;
    }
    printf("connector %d: %s (%s)\n",
           conn->connector_id,
           conn->connection == DRM_MODE_CONNECTED ? "connected" : "disconnected",
           conn->connector_id == 35 ? "HDMI-A-1" : "other");

    drmModeModeInfo *mode = get_preferred_mode(conn);
    if (!mode) {
        fprintf(stderr, "no mode available\n");
        drmModeFreeConnector(conn);
        close(fd);
        return 1;
    }
    printf("mode: %s %dx%d@%d\n", mode->name,
           mode->hdisplay, mode->vdisplay, mode->vrefresh);

    /* 找 CRTC */
    uint32_t crtc_id = find_crtc(fd, conn);
    if (!crtc_id) {
        fprintf(stderr, "no available CRTC\n");
        drmModeFreeConnector(conn);
        close(fd);
        return 1;
    }
    printf("crtc: %u\n", crtc_id);

    /* 创建 framebuffer */
    struct dumb_ctx fb = create_dumb_fb(fd, mode->hdisplay, mode->vdisplay);
    if (!fb.map || !fb.fb_id) {
        fprintf(stderr, "failed to create dumb framebuffer\n");
        drmModeFreeConnector(conn);
        close(fd);
        return 1;
    }
    printf("fb: id=%u pitch=%zu size=%zu\n", fb.fb_id, fb.pitch, fb.size);

    /* 填充图案 */
    fill_test_pattern(fb.map, mode->hdisplay, mode->vdisplay, fb.pitch);
    printf("test pattern drawn\n");

    /* Set CRTC — 关键步骤 */
    if (drmModeSetCrtc(fd, crtc_id, fb.fb_id, 0, 0,
                       &conn->connector_id, 1, mode) < 0) {
        fprintf(stderr, "drmModeSetCrtc failed: %s\n", strerror(errno));
        munmap(fb.map, fb.size);
        drmModeFreeConnector(conn);
        close(fd);
        return 1;
    }
    printf("drmModeSetCrtc OK — HDMI should now output test pattern\n");
    printf("Press Ctrl+C to stop\n");

    /* 保持运行 */
    while (keep_running) {
        sleep(1);
    }

    /* 清理 */
    drmModeSetCrtc(fd, crtc_id, 0, 0, 0, NULL, 0, NULL);
    drmModeFreeConnector(conn);
    munmap(fb.map, fb.size);
    close(fd);
    printf("cleanup done\n");
    return 0;
}
