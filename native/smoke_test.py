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

from animesr_host import ANIMESR_MODE_ID, ANISCALE2_MODE_ID, PROFILES


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
    parser.add_argument("--mode", choices=("animesr", "aniscale2"), default="animesr")
    args = parser.parse_args()
    width, height = args.width, args.height
    mode_id = ANISCALE2_MODE_ID if args.mode == "aniscale2" else ANIMESR_MODE_ID
    profile = PROFILES.get((mode_id, width, height))
    if not profile:
        raise ValueError("The selected native mode does not support this input size.")
    scale = profile[1]
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
            "type": "hello", "requestId": "smoke", "modeId": mode_id,
            "width": width, "height": height
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
        assert started == {"type": "started", "outputWidth": width * scale, "outputHeight": height * scale}

        frame = bytes(frame_bytes)
        received_segment = False
        for _ in range(6):
            websocket.send(frame)
            received_ready = False
            while not received_ready:
                message = websocket.receive(timeout=10)
                if isinstance(message, bytes):
                    received_segment = received_segment or len(message) > 0
                elif message:
                    response = json.loads(message)
                    if response.get("type") == "error":
                        raise RuntimeError(response.get("error"))
                    received_ready = response.get("type") == "ready"
            if received_segment:
                break

        assert received_segment, "Native host did not return an AV1/fMP4 segment."
        elapsed = time.perf_counter() - started_at
        print(f"{args.mode} native smoke test passed: {width}x{height} -> {width * scale}x{height * scale} in {elapsed:.2f}s")
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
