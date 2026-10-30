"""Streaming timeline serialization (schema v1 in PLAN §2.3).

Frames are written to a temp JSONL file during decode, then assembled into
the final schema-versioned JSON at the end — RAM stays flat for long videos.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction

SCHEMA_VERSION = "1.0.0"


def make_video_meta(filename: str, width: int, height: int, fps: float,
                    total_frames: int, duration_ms: float) -> dict:
    fr = Fraction(fps).limit_denominator(1000)
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "fps_num": fr.numerator,
        "fps_den": fr.denominator,
        "total_frames": total_frames,
        "duration_ms": round(duration_ms, 3),
        "timestamp_basis": "output_presentation_time",
    }


class FrameWriter:
    """Appends one JSON object per line. Call close(), then finalize()."""

    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        self.f = open(jsonl_path, "w", encoding="utf-8")
        self.count = 0

    def add(self, frame: dict) -> None:
        self.f.write(json.dumps(frame, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self) -> None:
        self.f.close()

    def finalize(self, json_path: str, header: dict) -> dict:
        """Assemble {header..., frames: [...]} from the JSONL lines.

        Returns the timeline as a dict (inline-able in a Cog dict output);
        also written to json_path for consumers that want the file.
        """
        frames = []
        with open(self.path, encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    frames.append(json.loads(line))
        os.remove(self.path)
        timeline = {**header, "frames": frames}
        with open(json_path, "w", encoding="utf-8") as out:
            json.dump(timeline, out, ensure_ascii=False)
        return timeline


__all__ = ["SCHEMA_VERSION", "FrameWriter", "make_video_meta", "load_timeline"]


def load_timeline(path: str) -> dict:
    """Read a timeline.json back into a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
