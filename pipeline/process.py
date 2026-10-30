"""End-to-end video -> (annotated MP4, timeline.json + inline timeline dict) pipeline."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import av
import numpy as np

from .annotate import annotate
from .describe import MotionTracker, describe, hex_to_hsv, hsv_close
from .detector import Detector
from .timeline import SCHEMA_VERSION, FrameWriter, make_video_meta
from .tracker import FrameTracker
from .video_io import (
    Writer,
    VideoError,
    extract_audio,
    finalize,
    mux_audio,
    probe,
    validate_video,
)


def _np(x):
    """torch.Tensor / ndarray -> ndarray."""
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _clip_bbox(bbox, width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    return [x1, y1, x2, y2]


def _class_summary(jsonl_path: str, display_labels: List[str]) -> List[dict]:
    """Per-class stats over all frames: count, avg/median/max confidence."""
    import statistics

    confs: dict = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            frame = json.loads(line)
            for o in frame["objects"]:
                # JSON objects use display names; map back via class_id for stats
                name = display_labels[o["class_id"]] if o.get("class_id") is not None else o["class_name"]
                confs.setdefault(name, []).append(o["confidence"])
    summary = []
    for name in display_labels:
        c = confs.get(name)
        if not c:
            continue
        summary.append({
            "class_name": name,
            "count": len(c),
            "avg_confidence": round(sum(c) / len(c), 4),
            "median_confidence": round(statistics.median(c), 4),
            "max_confidence": round(max(c), 4),
        })
    return summary


def process_video(
    input_video: str,
    workdir: str,
    model: Detector,
    classes: List[str],
    display_labels: List[str] = None,
    confidence: float = 0.25,
    iou_threshold: float = 0.7,
    tracker: str = "bytetrack",
    preserve_audio: bool = True,
    max_num_boxes: int = 10,
    color_filter: List = None,
) -> Tuple[str, str, dict]:
    """Returns (mp4_path, json_path, timeline_dict). Raises VideoError on invalid input.

    display_labels: optional renames shown in video/JSON (same order as classes);
                    detection always uses the precise `classes` prompts.
    """
    if display_labels is None:
        display_labels = list(classes)
    if color_filter is None:
        color_filter = []
    info = probe(input_video)
    validate_video(info)
    os.makedirs(workdir, exist_ok=True)
    raw_mp4 = os.path.join(workdir, "output_raw.mp4")
    json_path = os.path.join(workdir, "timeline.json")
    jsonl_tmp = os.path.join(workdir, ".frames.jsonl")

    model.set_classes(classes)
    # CFR policy (PLAN §6.2): keep input fps if <=30, else normalize to 30.
    fps = min(info.fps, 30.0)
    tracker_state = FrameTracker(tracker, fps)
    motion = MotionTracker()

    writer = Writer(raw_mp4, info.width, info.height, fps)
    fw = FrameWriter(jsonl_tmp)
    frames_written = 0

    try:
        with av.open(input_video) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]

                results = model.detect(img, conf=confidence, iou=iou_threshold)
                dets = np.empty((0, 4), dtype=float)
                confs = np.empty((0,), dtype=float)
                clss = np.empty((0,), dtype=int)
                if results and results[0].boxes is not None and len(results[0].boxes):
                    boxes = results[0].boxes
                    dets = _np(boxes.xyxy).reshape(-1, 4)
                    confs = _np(boxes.conf).reshape(-1)
                    clss = _np(boxes.cls).reshape(-1).astype(int)
                    # max_num_boxes: keep highest-confidence dets, stable order
                    if len(confs) > max_num_boxes:
                        keep = np.argsort(confs)[-max_num_boxes:]
                        dets, confs, clss = dets[keep], confs[keep], clss[keep]

                ids = tracker_state.update(frames_written, dets, confs, clss)

                objects = []
                for i in range(len(dets)):
                    bb = _clip_bbox(dets[i], w, h)
                    attrs = describe(img, bb)
                    if any(cf is not None for cf in color_filter) and not any(
                        hsv_close(attrs["hsv"], cf)
                        for cf in color_filter if cf is not None
                    ):
                        continue  # pixel color doesn't match any target color
                    tid = int(ids[i]) if ids[i] is not None else None
                    attrs["motion"] = motion.update(tid, bb)
                    objects.append(
                        {
                            "track_id": tid,
                            "class_id": int(clss[i]),
                            "class_name": display_labels[int(clss[i])],
                            **attrs,
                            "confidence": round(float(confs[i]), 4),
                            "bbox_xyxy": [round(v, 2) for v in bb],
                        }
                    )

                ts_ms = writer.write(annotate(img, objects))
                src_ms = float(frame.pts) * time_base * 1000.0 if frame.pts is not None else 0.0

                fw.add(
                    {
                        "frame_index": frames_written,
                        "timestamp_ms": round(ts_ms, 3),
                        "source_timestamp_ms": round(src_ms, 3),
                        "objects": objects,
                    }
                )
                frames_written += 1
    except av.error.InvalidDataError as e:
        fw.close()
        raise VideoError(f"decode failed mid-stream: {e}") from e

    writer.close()
    fw.close()

    duration_ms = frames_written / fps * 1000.0
    header = {
        "schema_version": SCHEMA_VERSION,
        "video": make_video_meta(
            "output.mp4", info.width, info.height, fps, frames_written, duration_ms
        ),
        "model": {
            "detector": f"yolov8{model.model_size}-worldv2",
            "tracker": tracker,
            "classes": list(classes),
            "display_names": list(display_labels),
            "confidence_threshold": confidence,
        },
        "summary": _class_summary(jsonl_tmp, display_labels),
    }
    timeline = fw.finalize(json_path, header)

    final_mp4 = os.path.join(workdir, "output.mp4")
    audio_path = os.path.join(workdir, "audio.m4a")
    if preserve_audio and info.has_audio:
        extract_audio(input_video, audio_path)
        mux_audio(raw_mp4, audio_path, final_mp4)
        os.remove(raw_mp4)
        os.remove(audio_path)
    else:
        finalize(raw_mp4)
        os.replace(raw_mp4, final_mp4)

    return final_mp4, json_path, timeline
