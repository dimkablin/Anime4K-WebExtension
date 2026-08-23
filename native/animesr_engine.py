"""TensorRT inference and AV1 encoding for AnimeSR v2."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

WIDTH = 1280
HEIGHT = 720
SCALE = 4
FRAME_BYTES = WIDTH * HEIGHT * 4


class AnimeSREngine:
    """Owns the recurrent TensorRT state for one 720p stream."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import torch

        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine was not found: {engine_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("AnimeSR requires an NVIDIA CUDA GPU.")

        self.torch = torch
        self.dtype = torch.float16
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not load the AnimeSR engine. Rebuild it for this GPU/runtime.")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self._allocate_buffers()
        self.reset()

    def _allocate_buffers(self) -> None:
        torch = self.torch
        device = torch.device("cuda")
        frame_shape = (1, 3, HEIGHT, WIDTH)
        output_shape = (1, 3, HEIGHT * SCALE, WIDTH * SCALE)
        state_shape = (1, 64, HEIGHT, WIDTH)

        self.prev_frame = torch.zeros(frame_shape, device=device, dtype=self.dtype).contiguous()
        self.curr_frame = torch.zeros(frame_shape, device=device, dtype=self.dtype).contiguous()
        self.next_frame = torch.zeros(frame_shape, device=device, dtype=self.dtype).contiguous()
        self.output = torch.zeros(output_shape, device=device, dtype=self.dtype).contiguous()
        self.state = torch.zeros(state_shape, device=device, dtype=self.dtype).contiguous()
        self.state_output = torch.zeros(state_shape, device=device, dtype=self.dtype).contiguous()

        buffers = {
            "prev_frame": self.prev_frame,
            "curr_frame": self.curr_frame,
            "next_frame": self.next_frame,
            "fb": self.output,
            "state": self.state,
            "out_img": self.output,
            "out_state": self.state_output,
        }
        for name, tensor in buffers.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
            if self.engine.get_tensor_mode(name).name == "INPUT":
                self.context.set_input_shape(name, tensor.shape)

    def reset(self) -> None:
        for tensor in (
            self.prev_frame,
            self.curr_frame,
            self.next_frame,
            self.output,
            self.state,
            self.state_output,
        ):
            tensor.zero_()
        self.pending = None
        self.first_run = True

    def push_rgba(self, rgba: bytes) -> bytes | None:
        frame = self._rgba_to_tensor(rgba)
        if self.pending is None:
            self.pending = frame
            return None

        output = self._infer(self.pending, frame)
        self.pending = frame
        rgb = output.clamp_(0, 1).mul_(255).to(self.torch.uint8)
        return rgb.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy().tobytes()

    def _rgba_to_tensor(self, rgba: bytes):
        if len(rgba) != FRAME_BYTES:
            raise ValueError(f"Expected {FRAME_BYTES} RGBA bytes, received {len(rgba)}")
        frame = self.torch.frombuffer(bytearray(rgba), dtype=self.torch.uint8)
        frame = frame.reshape(HEIGHT, WIDTH, 4)[:, :, :3]
        return frame.permute(2, 0, 1).unsqueeze(0).cuda().to(self.dtype).div_(255)

    def _infer(self, current, following):
        self.curr_frame.copy_(current)
        self.next_frame.copy_(following)
        if self.first_run:
            self.prev_frame.copy_(current)
            self.first_run = False

        with self.torch.cuda.stream(self.stream):
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed.")
        self.stream.synchronize()
        self.state.copy_(self.state_output)
        self.prev_frame.copy_(current)
        return self.output


class AV1Encoder:
    """Feeds native x4 RGB frames to NVENC and forwards fragmented MP4."""

    def __init__(self, fps: float, send_segment: Callable[[bytes], None]):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg with av1_nvenc is required in PATH.")
        self.send_segment = send_segment
        self.process = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{WIDTH * SCALE}x{HEIGHT * SCALE}",
                "-r", str(min(max(fps, 1.0), 60.0)), "-i", "pipe:0", "-an",
                "-pix_fmt", "yuv420p", "-c:v", "av1_nvenc",
                "-preset", "p1", "-tune", "ull", "-rc", "constqp", "-qp", "26",
                "-g", "1", "-bf", "0",
                "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                "-flush_packets", "1", "-f", "mp4", "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        threading.Thread(target=self._forward_output, daemon=True).start()

    def _forward_output(self) -> None:
        assert self.process.stdout is not None
        while chunk := self.process.stdout.read(256 * 1024):
            try:
                self.send_segment(chunk)
            except Exception:
                break

    def write(self, rgb: bytes) -> None:
        if self.process.poll() is not None:
            raise RuntimeError("AV1 encoder stopped unexpectedly.")
        assert self.process.stdin is not None
        self.process.stdin.write(rgb)

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
