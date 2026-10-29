"""Single-image path: annotated JPG + one-frame timeline.json."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import cv2
import numpy as np

from .annotate import annotate
from .describe import MotionTracker, describe, hex_to_hsv, hsv_close
from .timeline import SCHEMA_VERSION, make_video_meta
from .tracker import FrameTracker


def process_image(
    input_image: str,
    workdir: str,
    model,
    classes: List[str],
    display_labels: List[str] = None,
    confidence: float = 0.25,
    iou_threshold: float = 0.7,
    tracker: str = "bytetrack",
    max_num_boxes: int = 10,
    color_filter: List[str] = None,
) -> Tuple[str, str]:
    """Returns (jpg_path, json_path). No video decode/encode involved."""
    if display_labels is None:
        display_labels = list(classes)
    if color_filter is None:
        color_filter = []
    img = cv2.imread(input_image)
    if img is None:
        raise ValueError("cannot read image file")
    h, w = img.shape[:2]

    os.makedirs(workdir, exist_ok=True)
    jpg_path = os.path.join(workdir, "output.jpg")
    json_path = os.path.join(workdir, "timeline.json")

    model.set_classes(classes)
    results = model.detect(img, conf=confidence, iou=iou_threshold)
    ids = FrameTracker(tracker, 1)  # single frame: no motion history
    dets = np.empty((0, 4))
    confs = np.empty((0,))
    clss = np.empty((0,), dtype=int)
    if results and results[0].boxes is not None and len(results[0].boxes):
        boxes = results[0].boxes
        dets = boxes.xyxy.cpu().numpy().reshape(-1, 4)
        confs = boxes.conf.cpu().numpy().reshape(-1)
        clss = boxes.cls.cpu().numpy().reshape(-1).astype(int)
        if len(confs) > max_num_boxes:
            keep = np.argsort(confs)[-max_num_boxes:]
            dets, confs, clss = dets[keep], confs[keep], clss[keep]
    track_ids = ids.update(0, dets, confs, clss)
    mt = MotionTracker()

    objects = []
    for i in range(len(dets)):
        # clip bbox to image bounds
        x1, y1, x2, y2 = (float(v) for v in dets[i])
        bb = [
            max(0.0, min(x1, x2, w)), max(0.0, min(y1, y2, h)),
            max(0.0, min(x1, x2, w)), max(0.0, min(y1, y2, h)),
        ]
        bb[2] = min(w, max(x1, x2))
        bb[3] = min(h, max(y1, y2))
        bb = [bb[0], bb[1], bb[2], bb[3]]
        tid = track_ids[i]
        attrs = describe(img, bb)
        if any(cf is not None for cf in color_filter) and not any(
            hsv_close(attrs["hsv"], cf)
            for cf in color_filter if cf is not None
        ):
            continue  # pixel color doesn't match any target color
        attrs["motion"] = mt.update(tid, bb)
        objects.append({
            "track_id": tid,
            "class_id": int(clss[i]),
            "class_name": display_labels[int(clss[i])],
            **attrs,
            "confidence": round(float(confs[i]), 4),
            "bbox_xyxy": [round(v, 2) for v in bb],
        })

    cv2.imwrite(jpg_path, annotate(img, objects))

    timeline = {
        "schema_version": SCHEMA_VERSION,
        "video": make_video_meta("output.jpg", w, h, 1.0, 1, 0),
        "model": {
            "detector": f"yolov8{model.model_size}-worldv2",
            "tracker": tracker,
            "classes": list(classes),
            "display_names": list(display_labels),
            "confidence_threshold": confidence,
        },
        "summary": _class_summary(objects, display_labels),
        "frames": [{
            "frame_index": 0,
            "timestamp_ms": 0.0,
            "source_timestamp_ms": 0.0,
            "objects": objects,
        }],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False)
    return jpg_path, json_path
