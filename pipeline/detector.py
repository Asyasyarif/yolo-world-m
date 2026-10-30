"""YOLO-World open-vocabulary detector wrapper."""

from __future__ import annotations

from typing import List

import numpy as np

import os

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
SUPPORTED_SIZES = {"s", "m"}


def weights_path(model_size: str = "s") -> str:
    """weights/<size> must exist in the image (bundled at build time)."""
    p = os.path.join(WEIGHTS_DIR, f"yolov8{model_size}-worldv2.pt")
    if not os.path.exists(p):
        raise ValueError(
            f"model_size '{model_size}' not available: {p} not found, "
            f"choose from {sorted(SUPPORTED_SIZES)}"
        )
    return p


class Detector:
    """Thin wrapper around ultralytics YOLOWorld; custom text classes per run."""

    def __init__(self, model_size: str = "s"):
        import pathlib

        import ultralytics.nn.text_model as text_model
        from ultralytics import YOLOWorld

        self.model_size = model_size
        # ultralytics loads CLIP from <WEIGHTS_DIR>/clip, downloading there on
        # miss. Point that at our bundled weights so it never fetches 338MB.
        text_model.WEIGHTS_DIR = pathlib.Path(WEIGHTS_DIR)
        self.model = YOLOWorld(weights_path(model_size))

    def set_classes(self, labels: List[str]) -> None:
        self.model.set_classes(labels)
        self.labels = labels

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.25,
               iou: float = 0.7) -> np.ndarray:
        """Run detection on one BGR frame. Returns ultralytics Results list."""
        return self.model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)


__all__ = ["Detector", "SUPPORTED_SIZES", "weights_path"]
