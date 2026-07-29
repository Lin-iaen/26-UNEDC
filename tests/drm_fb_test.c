/*
 * drm_fb_test — 用 libdrm 分配 dumb buffer → drmModeSetCrtc
 *
 * 编译:
 *   gcc -o tests/drm_fb_test tests/drm_fb_test.c -ldrm
 *
 * 用法:
 *   模式 1 — 静态测试图案:
 *     ./tests/drm_fb_test <DRM_DEV> <CONNECTOR_ID>
 *     显示 SMPTE 彩条棋盘格，按 Ctrl+C 退出。
 *
 *   模式 2 — 从 stdin 推流:
 *     ./tests/drm_fb_test <DRM_DEV> <CONNECTOR_ID> stream <W> <H>
 *     从 stdin 读取 BGR24 帧 (W*H*3 字节)，写入 dumb buffer 实现实时显示。
 *
 *   典型自环采集流程:
 *     # 1. 后台 UVC 流唤醒 HDMI
 *     ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 \
 *            -i /dev/video8 -t 30 -f null - &
 *     sleep 4
 *     # 2. 运行本程序（摄像头画面通过 stdin 传入）
 *     python record_loopback.py   # 内部调用本程序 + ffmpeg
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

    /* Set CRTC — 关键步骤 */
    if (drmModeSetCrtc(fd, crtc_id, fb.fb_id, 0, 0,
                       &conn->connector_id, 1, mode) < 0) {
        fprintf(stderr, "drmModeSetCrtc failed: %s\n", strerror(errno));
        munmap(fb.map, fb.size);
        drmModeFreeConnector(conn);
        close(fd);
        return 1;
    }
    fprintf(stderr, "drmModeSetCrtc OK — HDMI active\n");

    /* 判断模式 */
    int is_stream = (argc > 3 && strcmp(argv[3], "stream") == 0);
    int stream_w = is_stream && argc > 4 ? atoi(argv[4]) : mode->hdisplay;
    int stream_h = is_stream && argc > 5 ? atoi(argv[5]) : mode->vdisplay;
    int fb_w = is_stream ? stream_w : mode->hdisplay;
    int fb_h = is_stream ? stream_h : mode->vdisplay;

    if (is_stream) {
        /* 在流模式下使用自定义分辨率创建 framebuffer */
        /* 先释放之前为 preferred mode 创建的 fb */
        munmap(fb.map, fb.size);
        drmModeRmFB(fd, fb.fb_id);
        /* 创建匹配 stream 尺寸的 dumb buffer */
        struct drm_mode_destroy_dumb dd = {.handle = fb.handle};
        drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);

        fb = create_dumb_fb(fd, fb_w, fb_h);
        if (!fb.map || !fb.fb_id) {
            fprintf(stderr, "failed to create stream-sized dumb buffer\n");
            goto cleanup;
        }
        fprintf(stderr, "stream fb: %dx%d id=%u pitch=%zu\n",
                fb_w, fb_h, fb.fb_id, fb.pitch);

        /* 构建 640x480@60 的 mode info */
        drmModeModeInfo stream_mode = {
            .clock = 25175,
            .hdisplay = fb_w, .hsync_start = fb_w + 16,
            .hsync_end = fb_w + 16 + 48, .htotal = fb_w + 16 + 48 + 96,
            .vdisplay = fb_h, .vsync_start = fb_h + 10,
            .vsync_end = fb_h + 10 + 2, .vtotal = fb_h + 10 + 2 + 33,
            .vrefresh = 60,
            .flags = DRM_MODE_FLAG_NHSYNC | DRM_MODE_FLAG_NVSYNC,
            .type = DRM_MODE_TYPE_DRIVER,
            .name = "",
        };
        snprintf(stream_mode.name, sizeof(stream_mode.name),
                 "%dx%d", fb_w, fb_h);

        if (drmModeSetCrtc(fd, crtc_id, fb.fb_id, 0, 0,
                           &conn->connector_id, 1, &stream_mode) < 0) {
            fprintf(stderr, "drmModeSetCrtc (stream) failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
        fprintf(stderr, "stream mode set: %dx%d@60\n", fb_w, fb_h);
        /* 模式 2: 从 stdin 读 BGR24 帧 → 写入 dumb buffer */
        fprintf(stderr, "stream mode: %dx%d BGR24 from stdin\n",
                stream_w, stream_h);
        size_t frame_bytes = (size_t)stream_w * stream_h * 3;
        uint8_t *bgr = malloc(frame_bytes);
        if (!bgr) {
            fprintf(stderr, "malloc failed\n");
            goto cleanup;
        }

        while (keep_running) {
            size_t total = 0;
            while (total < frame_bytes && keep_running) {
                ssize_t n = read(STDIN_FILENO, bgr + total,
                                 frame_bytes - total);
                if (n <= 0) {
                    if (total == 0) {
                        /* 无数据 — 短暂等待后重试 */
                        usleep(5000);
                        break;
                    }
                    /* 部分数据 — 丢弃不完整帧 */
                    keep_running = 0;
                    break;
                }
                total += n;
            }
            if (total < frame_bytes) {
                if (keep_running)
                    continue;  /* 重试读帧 */
                break;
            }

            /* BGR24 → XRGB8888 逐行写入 dumb buffer */
            for (int y = 0; y < stream_h && y < fb_h; y++) {
                uint32_t *row = (uint32_t *)(fb.map + y * fb.pitch);
                int src_off = y * stream_w * 3;
                for (int x = 0; x < stream_w && x < fb_w; x++) {
                    uint8_t b = bgr[src_off + x * 3];
                    uint8_t g = bgr[src_off + x * 3 + 1];
                    uint8_t r = bgr[src_off + x * 3 + 2];
                    row[x] = (0xFF << 24) | (r << 16) | (g << 8) | b;
                }
            }

            /* 强制写入像素 (0,0) 为红色做标记 */
            uint32_t *p0 = (uint32_t *)(fb.map);
            p0[0] = 0xFFFF0000;  /* XRGB: B=0, G=0, R=255 */

            /* 通知 DRM 刷新 */
            drmModeDirtyFB(fd, fb.fb_id, NULL, 0);

            static int frame_count = 0;
            frame_count++;
            if (frame_count % 30 == 0) {
                fprintf(stderr, "  stream: %d frames\n", frame_count);
            }
        }
        free(bgr);
    } else {
        /* 模式 1: 静态测试图案 */
        fill_test_pattern(fb.map, fb_w, fb_h, fb.pitch);
        fprintf(stderr, "test pattern drawn\n");
        fprintf(stderr, "Press Ctrl+C to stop\n");

        while (keep_running) {
            sleep(1);
        }
    }

cleanup:

    /* 清理 */
    drmModeSetCrtc(fd, crtc_id, 0, 0, 0, NULL, 0, NULL);
    drmModeFreeConnector(conn);
    munmap(fb.map, fb.size);
    close(fd);
    printf("cleanup done\n");
    return 0;
}
