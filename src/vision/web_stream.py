"""Pluggable MJPEG HTTP streaming engine.

Pure infrastructure component — contains zero business logic or hardware
coupling.  Receives frames via a caller-supplied ``frame_provider`` callable
and serves them to web clients.

Usage::

    from src.vision.web_stream import MjpegStreamer
    from src.drivers import Camera

    cam = Camera()
    cam.start()

    streamer = MjpegStreamer(frame_provider=cam.read, port=5000)
    streamer.start()

    # … main loop …
    streamer.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
from flask import Flask, Response, make_response, render_template_string

logger = logging.getLogger("vision.web_stream")

DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MJPEG Stream</title>
    <style>
        body { margin: 0; background: #000; display: flex;
               align-items: center; justify-content: center;
               min-height: 100vh; }
        img { max-width: 100%; height: auto; display: block; }
    </style>
</head>
<body>
    <img src="/video_feed">
</body>
</html>"""


class MjpegStreamer:
    """Background MJPEG HTTP server driven by an injected frame provider.

    **No business logic** — this class only moves pixels to browsers.
    """

    def __init__(
        self,
        frame_provider: Callable[[], np.ndarray | None],
        port: int = 5000,
        custom_template: str | None = None,
        custom_routes: dict[str, Callable[[], Any]] | None = None,
    ) -> None:
        """Initialise but do **not** start the server.

        Args:
            frame_provider: Zero-argument callable returning a BGR ``(H,W,3)``
                ``np.ndarray``, or ``None`` if no frame is ready.
            port: TCP port to bind.
            custom_template: Optional HTML string to serve at ``/``.  Pass
                ``None`` to use a built-in minimal full-screen ``<img>`` page.
            custom_routes: Optional ``{path: handler}`` dict.  Each handler is
                registered via ``app.add_url_rule(path, ...)`` and must
                accept the standard Flask view signature ``(**kwargs)``.
        """
        self._frame_provider = frame_provider
        self._port = port
        self._template = custom_template or DEFAULT_TEMPLATE
        self._custom_routes = custom_routes or {}

        self._app = Flask(__name__)
        self._thread: threading.Thread | None = None
        self._running = False

        self._register_routes()

    # ── public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch Flask in a daemon thread (non-blocking)."""
        if self._running:
            return

        # Silence Werkzeug HTTP request logs
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info("Streamer started on port %d", self._port)

    def stop(self) -> None:
        """Signal the server thread to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Streamer stopped")

    # ── routes ────────────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        """Wire up the built-in routes and any caller-supplied extras."""

        @self._app.route("/")
        def index():
            response = make_response(render_template_string(self._template))
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

        @self._app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        for path, handler in self._custom_routes.items():
            # Generate a unique endpoint name from the path
            endpoint = path.lstrip("/").replace("/", "_").replace("<", "_").replace(">", "_")
            def make_view(h=handler):
                def view(**kwargs):
                    return h(**kwargs)
                return view
            self._app.add_url_rule(path, endpoint=endpoint, view_func=make_view())

    # ── internals ─────────────────────────────────────────────────────────

    def _generate_frames(self):
        """Generator: pull a frame, JPEG-encode, emit MJPEG boundary.

        JPEG compression runs **only** when a client is connected to
        ``/video_feed`` — idle time costs zero CPU.
        """
        while self._running:
            frame = self._frame_provider()
            if frame is None:
                time.sleep(0.05)
                continue

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(0.01)  # ~100 fps ceiling

    def _serve(self) -> None:
        """Internal: block on ``app.run()`` until ``_running`` is cleared."""
        try:
            self._app.run(
                host="0.0.0.0",
                port=self._port,
                threaded=True,
                use_reloader=False,
            )
        except Exception:
            logger.exception("Flask server crashed")
        finally:
            self._running = False
