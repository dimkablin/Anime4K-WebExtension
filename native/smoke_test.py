"""One-frame end-to-end check for the native host and AV1 stream."""

from __future__ import annotations

import json
import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from simple_websocket import Client, ConnectionClosed

from animesr_engine import SCALE
from animesr_host import ENGINE_NAMES


def send_native(process: subprocess.Popen, message: dict) -> None:
    payload = json.dumps(message).encode()
    assert process.stdin is not None
    process.stdin.write(struct.pack("=I", len(payload)) + payload)
    process.stdin.flush()


def receive_native(process: subprocess.Popen) -> dict:
    assert process.stdout is not None
    size = struct.unpack("=I", process.stdout.read(4))[0]
    return json.loads(process.stdout.read(size))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    width, height = args.width, args.height
    if (width, height) not in ENGINE_NAMES:
        raise ValueError("Smoke test supports 1280x720 and 1920x1080.")
    frame_bytes = width * height * 4

    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.setdefault("ANIMESR_MODEL_DIR", str(root / "models"))
    process = subprocess.Popen(
        [sys.executable, str(root / "animesr_host.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    websocket = None
    started_at = time.perf_counter()
    try:
        send_native(process, {
            "type": "hello", "requestId": "smoke", "width": width, "height": height
        })
        native = receive_native(process)
        if not native.get("ok"):
            raise RuntimeError(native.get("error", "Native host handshake failed."))

        websocket = Client.connect(native["endpoint"], max_message_size=frame_bytes)
        websocket.send(json.dumps({"type": "start", "width": width, "height": height, "fps": 24}))
        started_message = websocket.receive(timeout=60)
        if started_message is None:
            raise TimeoutError("AnimeSR session did not start within 60 seconds.")
        started = json.loads(started_message)
        assert started == {"type": "started", "outputWidth": width * SCALE, "outputHeight": height * SCALE}

        frame = bytes(frame_bytes)
        websocket.send(frame)
        assert json.loads(websocket.receive(timeout=10))["type"] == "ready"
        websocket.send(frame)

        received_segment = False
        received_ready = False
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (received_segment and received_ready):
            message = websocket.receive(timeout=2)
            if isinstance(message, bytes):
                received_segment = received_segment or len(message) > 0
            elif message:
                response = json.loads(message)
                if response.get("type") == "error":
                    raise RuntimeError(response.get("error"))
                received_ready = received_ready or response.get("type") == "ready"

        assert received_ready, "Native host did not acknowledge the inferred frame."
        assert received_segment, "Native host did not return an AV1/fMP4 segment."
        elapsed = time.perf_counter() - started_at
        print(f"AnimeSR native smoke test passed: {width}x{height} -> {width * SCALE}x{height * SCALE} in {elapsed:.2f}s")
        return 0
    finally:
        if websocket:
            try:
                websocket.close()
            except ConnectionClosed:
                pass
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stderr:
            error_output = process.stderr.read().decode(errors="replace").strip()
            if error_output:
                print(error_output, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
