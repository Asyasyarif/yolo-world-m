"""Visual descriptors per detection: color + shape + size (+ motion elsewhere).

All computed from pixels/geometry — no ML. Prompt-based attribute requests
("red ball") measurably hurt YOLO-World, so attributes live as separate
metadata fields instead.
"""

from __future__ import annotations

import cv2
import numpy as np

# (name, center hue) — hue in OpenCV's 0-179 range; red wraps at 0/180
NAMED_HUES = [
    (5.0, "red"),
    (15.0, "orange"),
    (30.0, "yellow"),
    (60.0, "green"),
    (90.0, "cyan"),
    (120.0, "blue"),
    (155.0, "purple"),
]


def hex_to_hsv(hex_str: str) -> tuple:
    """'#rrggbb' -> (h, s, v) in OpenCV scale (h 0-180, s/v 0-255)."""
    if not (hex_str.startswith("#") and len(hex_str) == 7):
        raise ValueError(
            f"color '{hex_str}' not understood; use hex like #ff0000, "
            "or rgb() / hsl() names"
        )
    r = int(hex_str[1:3], 16)
    g = int(hex_str[3:5], 16)
    b = int(hex_str[5:7], 16)
    h, s, v = cv2.cvtColor(
        np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV
    )[0, 0]
    return float(h), float(s), float(v)


def hsv_close(med: tuple, target: tuple, hue_tol: float = 25.0) -> bool:
    """Does the detected median HSV 'look like' the target color?

    ponytail: fixed tolerances (25° hue, 70 value, 50 sat) — tune if real
    footage proves harsher. Value-only match for gray targets; saturated
    targets demand saturated pixels.
    """
    if med is None or med[0] < 0:
        return False
    mh, ms, mv = med
    th, ts, tv = target
    if ts < 50:  # target gray/white/black: ignore hue
        return abs(mv - tv) < 70 and ms < 60  # pixels must be near-neutral
    if ms < 50:  # observed pixel unsaturated — can't be a saturated target
        return False
    dh = abs(mh - th)
    dh = min(dh, 180 - dh)  # angular: hue wraps
    return dh <= hue_tol


def _color(img_bgr, bbox) -> str:
    med_hue, med_sat, med_val = _hsv_stats(img_bgr, bbox)
    if med_val < 50:
        return "black"
    if med_sat < 40:
        return "white" if med_val > 180 else "gray"
    centered = (med_hue + 10) % 180  # red wraps around 0
    return min(NAMED_HUES, key=lambda nh: abs(centered - nh[0]))[1]


def _hsv_stats(img_bgr, bbox):
    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return -1.0, -1.0, -1.0
    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    crop = crop[ch // 5: ch - ch // 5, cw // 5: cw - cw // 5]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return (
        float(np.median(hsv[..., 0].astype(float))),
        float(np.median(hsv[..., 1].astype(float))),
        float(np.median(hsv[..., 2].astype(float))),
    )


def _shape(crop_gray: np.ndarray) -> str:
    """Largest contour's circularity/aspect -> round/elongated/irregular."""
    _, bw = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    c = max(contours, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    if per == 0 or area == 0:
        return "unknown"
    circularity = 4 * np.pi * area / (per * per)
    _, _, w, h = cv2.boundingRect(c)
    aspect = max(w, h) / max(1, min(w, h))
    if circularity > 0.65 and aspect < 1.25:
        return "round"
    if aspect > 1.6:
        return "elongated"
    return "boxy"


def _size(w: int, h: int, frame_w: int, frame_h: int) -> str:
    frac = (w * h) / (frame_w * frame_h)
    if frac < 0.005:
        return "small"
    if frac < 0.05:
        return "medium"
    return "large"


def describe(img_bgr, bbox_xyxy) -> dict:
    """Attributes dict from one detection: color, shape, size, hsv."""
    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
    x2c, y2c = min(w_img, x2), min(h_img, y2)
    x1c, y1c = max(0, x1), max(0, y1)
    if x2c - x1c < 4 or y2c - y1c < 4:
        return {"color": "unknown", "shape": "unknown",
                "size": _size(x2 - x1, y2 - y1, w_img, h_img),
                "hsv": None}
    crop = img_bgr[y1c:y2c, x1c:x2c]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, s, v = _hsv_stats(img_bgr, bbox_xyxy)
    return {
        "color": _color_from(h, s, v),
        "shape": _shape(gray),
        "size": _size(x2 - x1, y2 - y1, w_img, h_img),
        "hsv": [round(h, 1), round(s, 1), round(v, 1)],
    }


def _color_from(h, s, v):
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 180 else "gray"
    centered = (h + 10) % 180
    return min(NAMED_HUES, key=lambda nh: abs(centered - nh[0]))[1]


class MotionTracker:
    """track_id -> last centroid; returns per-det motion dict."""

    def __init__(self):
        self.last: dict = {}

    def update(self, track_id, bbox):
        if track_id is None:
            return None
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        prev = self.last.get(track_id)
        self.last[track_id] = (cx, cy)
        if prev is None:
            return {"moving": False, "speed_px_frame": 0.0, "direction": None}
        dx, dy = cx - prev[0], cy - prev[1]
        speed = float(np.hypot(dx, dy))
        moving = speed > 2.0
        import math
        deg = None
        if moving:
            # screen coords: 0 = right, 90 = down
            deg = int(round(math.degrees(math.atan2(dy, dx)))) % 360
        return {"moving": moving, "speed_px_frame": round(speed, 2),
                "direction": deg}


__all__ = ["describe", "MotionTracker"]
