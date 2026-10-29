"""Draws the same objects that the timeline records onto video frames.

Color is constant per class name — every instance of "soccer ball" gets
the same color across frames, tracks, and predictions.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

# 16 distinct colors (BGR) — stable per class_id at any classes size.
PALETTE = [
    (56, 186, 60),    # green
    (250, 173, 20),   # amber
    (47, 138, 239),   # blue
    (214, 61, 46),    # red
    (155, 60, 180),   # purple
    (0, 200, 255),    # yellow
    (0, 128, 0),      # teal-dark
    (180, 105, 255),  # pink
    (42, 42, 165),    # orange-brown
    (100, 200, 100),  # light green
    (255, 255, 150),  # sky
    (80, 80, 80),     # gray
    (255, 0, 128),    # magenta
    (200, 240, 255),  # cream
    (0, 255, 128),    # spring
    (128, 128, 255),  # mauve
]
THICKNESS = 2
FONT_SCALE = 0.5


def stable_color(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


def _label(obj: Dict, show_confidence: bool, show_track_id: bool) -> str:
    parts = [obj["class_name"]]
    if show_track_id and obj.get("track_id") is not None:
        parts.append(f"#{obj['track_id']}")
    if show_confidence:
        parts.append(f": {obj['confidence']*100:.0f}%")
    return " ".join(parts)


def annotate(
    frame: np.ndarray,
    objects: List[Dict],
    show_confidence: bool = True,
    show_track_id: bool = True,
) -> np.ndarray:
    """Draw bboxes + labels for the objects dict-lists from process_video."""
    for obj in objects:
        x1, y1, x2, y2 = (int(v) for v in obj["bbox_xyxy"])
        # constant per class — class_id indexes the user's classes list.
        # Multiple classes can collide in the 4-entry palette; list order wins.
        color = stable_color(obj.get("class_id", 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
        label = _label(obj, show_confidence, show_track_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        ty = y1 - 6 if y1 - th - 6 > 0 else y1 + 2
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


__all__ = ["annotate", "stable_color"]
