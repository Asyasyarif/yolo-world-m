"""Video probing, decoding and encoding for the pipeline."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import av
import numpy as np

# ponytail: fixed MVP limits from PLAN § 6.1 — tune after benchmarking.
MAX_DURATION_S = 60.0
# Orientation-agnostic: limit the LONGER side to 1920, shorter to 1080.
MAX_LONG_SIDE = 1920
MAX_SHORT_SIDE = 1080
OUTPUT_FPS = 30  # CFR normalization target


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float
    has_video: bool
    has_audio: bool


class VideoError(ValueError):
    """Raised for invalid or out-of-limit input videos."""


def probe(video_path: str) -> VideoInfo:
    try:
        with av.open(video_path) as container:
            vstreams = container.streams.video
            astreams = container.streams.audio
            if not vstreams:
                raise VideoError("no video stream found in input file")
            vs = vstreams[0]
            if len(vstreams) > 1:
                raise VideoError("input has multiple video streams")
            fps = float(vs.average_rate) if vs.average_rate else 0.0
            if container.duration is not None:
                duration = container.duration / av.time_base
            else:
                # fall back to stream duration (in its own time_base)
                duration = (
                    float(vs.duration * vs.time_base) if vs.duration else 0.0
                )
            if fps <= 0 or duration <= 0:
                raise VideoError("cannot determine fps/duration of input video")
            return VideoInfo(
                width=vs.width,
                height=vs.height,
                fps=fps,
                duration_s=duration,
                has_video=True,
                has_audio=len(astreams) > 0,
            )
    except av.error.InvalidDataError as e:
        raise VideoError(f"invalid video data: {e}") from e
    except av.error.FailedToOpenError as e:
        raise VideoError(f"cannot open video file: {e}") from e


def validate_video(info: VideoInfo) -> None:
    if info.duration_s > MAX_DURATION_S:
        raise VideoError(
            f"video is {info.duration_s:.1f}s, limit is {MAX_DURATION_S}s"
        )
    long, short = max(info.width, info.height), min(info.width, info.height)
    if long > MAX_LONG_SIDE or short > MAX_SHORT_SIDE:
        raise VideoError(
            f"resolution {info.width}x{info.height} exceeds "
            f"{MAX_LONG_SIDE}x{MAX_SHORT_SIDE} (longest side rule)"
        )


class Writer:
    """H.264 CFR MP4 encoder. Decoded-and-annotated frames go in, one file out."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream("h264", rate=int(round(fps)))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "23"}
        self.fps = fps
        self.count = 0

    def write(self, frame: np.ndarray) -> float:
        """Write one BGR frame; returns its output presentation time in ms."""
        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
        ts = self.count / self.fps * 1000.0
        self.count += 1
        return ts

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def extract_audio(input_path: str, output_path: str) -> None:
    """Re-encode the input's first audio stream to AAC. VideoError if no audio."""
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", input_path,
        "-map", "0:a:0",
        "-c:a", "aac",
        "-b:a", "160k",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"audio extraction failed: {proc.stderr.splitlines()[-1]}")


def mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Copy the encoded video stream and add the audio track."""
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"audio mux failed: {proc.stderr.splitlines()[-1]}")


def finalize(video_path: str) -> None:
    """Add faststart so web playback does not wait on the full file."""
    tmp = video_path + ".faststart.mp4"
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", video_path,
        "-c", "copy",
        "-movflags", "+faststart",
        tmp,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"faststart remux failed: {proc.stderr.splitlines()[-1]}")
    subprocess.run(["mv", tmp, video_path], check=True)


__all__ = [
    "Writer",
    "VideoError",
    "VideoInfo",
    "extract_audio",
    "finalize",
    "mux_audio",
    "probe",
    "validate_video",
]
