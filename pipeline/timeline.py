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

    def finalize(self, json_path: str, header: dict) -> None:
        """Assemble {header..., frames: [...]} by streaming the JSONL lines."""
        with open(json_path, "w", encoding="utf-8") as out:
            out.write(json.dumps(header, ensure_ascii=False)[:-1])  # strip closing }
            out.write(', "frames": [')
            with open(self.path, encoding="utf-8") as lines:
                for i, line in enumerate(lines):
                    if i:
                        out.write(", ")
                    out.write(line.strip())
            out.write("]}\n")
        os.remove(self.path)


__all__ = ["SCHEMA_VERSION", "FrameWriter", "make_video_meta"]
