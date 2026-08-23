"""Chrome/Edge Native Messaging host for AnimeSR v2 TensorRT."""

from __future__ import annotations

import json
import os
import secrets
import struct
import sys
import threading
from pathlib import Path

WIDTH = 1280
HEIGHT = 720
SCALE = 4
MAX_NATIVE_MESSAGE = 1024 * 1024


def read_native_message(stream) -> dict | None:
    size_bytes = stream.read(4)
    if not size_bytes:
        return None
    if len(size_bytes) != 4:
        raise ValueError("Incomplete Native Messaging header.")
    size = struct.unpack("=I", size_bytes)[0]
    if size > MAX_NATIVE_MESSAGE:
        raise ValueError("Native Messaging request is too large.")
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("Incomplete Native Messaging payload.")
    return json.loads(payload.decode("utf-8"))


def write_native_message(stream, message: dict) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_NATIVE_MESSAGE:
        raise ValueError("Native Messaging response is too large.")
    stream.write(struct.pack("=I", len(payload)))
    stream.write(payload)
    stream.flush()


class MediaSession:
    def __init__(self, websocket, engine, fps: float):
        from animesr_engine import AV1Encoder

        self.websocket = websocket
        self.send_lock = threading.Lock()
        self.engine = engine
        self.engine.reset()
        self.encoder = AV1Encoder(fps, self.send_binary)

    def send_json(self, message: dict) -> None:
        with self.send_lock:
            self.websocket.send(json.dumps(message, separators=(",", ":")))

    def send_binary(self, payload: bytes) -> None:
        with self.send_lock:
            self.websocket.send(payload)

    def push(self, rgba: bytes) -> None:
        output = self.engine.push_rgba(rgba)
        if output is not None:
            self.encoder.write(output)
        self.send_json({"type": "ready"})

    def reset(self) -> None:
        self.engine.reset()
        self.send_json({"type": "ready"})

    def close(self) -> None:
        self.encoder.close()


class MediaServer:
    def __init__(self, engine_path: Path):
        from animesr_engine import AnimeSREngine
        from werkzeug.serving import WSGIRequestHandler, make_server

        self.engine_path = engine_path
        # CUDA/TensorRT must be initialized on the native host's main thread.
        self.engine = AnimeSREngine(engine_path)
        self.session_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)

        class QuietHandler(WSGIRequestHandler):
            def log(self, message: str, *args) -> None:
                pass

        self.server = make_server(
            "127.0.0.1", 0, self._application, threaded=True, request_handler=QuietHandler
        )
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def endpoint(self) -> str:
        return f"ws://127.0.0.1:{self.port}/{self.token}"

    def _application(self, environ, start_response):
        if environ.get("PATH_INFO") != f"/{self.token}":
            start_response("403 Forbidden", [("Content-Type", "text/plain")])
            return [b"Forbidden"]

        from simple_websocket import ConnectionClosed, Server

        websocket = Server.accept(environ, max_message_size=WIDTH * HEIGHT * 4)
        with self.session_lock:
            self._serve_websocket(websocket)
        return []

    def _serve_websocket(self, websocket) -> None:
        from simple_websocket import ConnectionClosed

        session = None
        try:
            print("AnimeSR media client connected.", file=sys.stderr, flush=True)
            message = websocket.receive(timeout=10)
            print(f"AnimeSR media start message received: {type(message).__name__}", file=sys.stderr, flush=True)
            config = json.loads(message) if isinstance(message, str) else {}
            if config.get("type") != "start":
                raise ValueError("The first media message must start a session.")
            if (int(config.get("width", 0)), int(config.get("height", 0))) != (WIDTH, HEIGHT):
                raise ValueError("AnimeSR currently requires 1280x720 input.")

            session = MediaSession(websocket, self.engine, float(config.get("fps", 24)))
            print("AnimeSR TensorRT session initialized.", file=sys.stderr, flush=True)
            session.send_json({
                "type": "started",
                "outputWidth": WIDTH * SCALE,
                "outputHeight": HEIGHT * SCALE,
            })
            while True:
                message = websocket.receive()
                if isinstance(message, bytes):
                    session.push(message)
                    continue
                command = json.loads(message)
                if command.get("type") == "reset":
                    session.reset()
        except ConnectionClosed:
            pass
        except Exception as error:
            print(f"AnimeSR media session failed: {error}", file=sys.stderr, flush=True)
            try:
                websocket.send(json.dumps({"type": "error", "error": str(error)}))
            except Exception:
                pass
        finally:
            if session:
                session.close()

    def close(self) -> None:
        self.server.shutdown()


def resolve_engine_path() -> Path:
    model_dir = Path(os.environ.get("ANIMESR_MODEL_DIR", Path(__file__).with_name("models")))
    return model_dir / "AnimeSR_v2_fp16_op20_fp16_720x1280.engine"


def self_test() -> None:
    import io

    stream = io.BytesIO()
    expected = {"type": "hello", "requestId": "test"}
    write_native_message(stream, expected)
    stream.seek(0)
    assert read_native_message(stream) == expected
    assert SCALE == 4 and WIDTH * SCALE == 5120 and HEIGHT * SCALE == 2880


def main() -> int:
    if "--self-test" in sys.argv:
        self_test()
        return 0

    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    media_server = None
    try:
        while request := read_native_message(sys.stdin.buffer):
            request_id = request.get("requestId")
            try:
                if request.get("type") != "hello":
                    raise ValueError("Unknown native host request.")
                if media_server is None:
                    media_server = MediaServer(resolve_engine_path())
                response = {"requestId": request_id, "ok": True, "endpoint": media_server.endpoint}
            except Exception as error:
                response = {"requestId": request_id, "ok": False, "error": str(error)}
            write_native_message(sys.stdout.buffer, response)
    finally:
        if media_server:
            media_server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
