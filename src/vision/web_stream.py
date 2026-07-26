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
from werkzeug.serving import make_server

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
        img { max-width: 100%; max-height: 100vh; object-fit: contain; display: block; }
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
        max_fps: float = 30.0,
        jpeg_quality: int = 75,
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
            max_fps: Ceiling on how often each connected client is served a
                frame.  Without it the generator spins as fast as it can and
                re-encodes the *same* frame several times per capture — pure
                waste that becomes crippling at high resolutions.  Set it at or
                slightly above the camera's capture rate.
            jpeg_quality: 1–100.  Lower values cut encode time and bandwidth.
        """
        self._frame_provider = frame_provider
        self._port = port
        self._template = custom_template or DEFAULT_TEMPLATE
        self._custom_routes = custom_routes or {}
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._jpeg_quality = int(jpeg_quality)

        self._app = Flask(__name__)
        self._thread: threading.Thread | None = None
        self._server: Any = None
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
        """Shut the HTTP server down and join the server thread."""
        # Clearing this first ends any in-flight _generate_frames loop, so the
        # streaming response finishes and shutdown() is not left waiting on it.
        self._running = False
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                logger.exception("Server shutdown failed")
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Streamer stopped")

    # ── routes ────────────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        """Wire up the built-in routes and any caller-supplied extras."""

        # Accept any method on the page itself.  A browser reload of a stalled
        # page can re-issue the original request as something other than GET,
        # and answering that with "405 Method Not Allowed" makes the UI look
        # permanently dead when the server is in fact fine.
        @self._app.route("/", methods=["GET", "POST", "HEAD", "OPTIONS"])
        def index():
            response = make_response(render_template_string(self._template))
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

        @self._app.route("/video_feed", methods=["GET", "POST", "HEAD"])
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
            self._app.add_url_rule(
                path,
                endpoint=endpoint,
                view_func=make_view(),
                methods=["GET", "POST"],
            )

    # ── internals ─────────────────────────────────────────────────────────

    def _generate_frames(self):
        """Generator: pull a frame, JPEG-encode, emit MJPEG boundary.

        JPEG compression runs **only** when a client is connected to
        ``/video_feed`` — idle time costs zero CPU.
        """
        while self._running:
            started = time.perf_counter()

            # A provider or encoder failure must not kill the generator — that
            # would tear down the MJPEG response and leave the browser staring
            # at a dead image until the page is reloaded.
            try:
                frame = self._frame_provider()
            except Exception:
                logger.exception("frame_provider raised")
                time.sleep(0.1)
                continue

            if frame is None:
                # Short poll: a provider that returns None between captures (to
                # avoid re-sending a frame the client already has) would other-
                # wise be sampled more coarsely than the camera produces frames.
                time.sleep(0.005)
                continue

            ok, jpeg = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            if not ok:
                logger.warning("JPEG encode failed (shape=%s)", getattr(frame, "shape", None))
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

            # Pace to max_fps.  Sleep only for the time the encode did not
            # already consume, so a slow encode is not penalised twice.
            spare = self._min_interval - (time.perf_counter() - started)
            if spare > 0:
                time.sleep(spare)

    def _serve(self) -> None:
        """Internal: serve requests until :meth:`stop` shuts the server down.

        Uses ``make_server`` rather than ``app.run()`` because the latter offers
        no way to shut down from another thread — ``stop()`` could only ever
        leave the server running until process exit.
        """
        try:
            self._server = make_server("0.0.0.0", self._port, self._app, threaded=True)
            self._server.serve_forever()
        except Exception:
            logger.exception("Flask server crashed")
        finally:
            self._running = False
