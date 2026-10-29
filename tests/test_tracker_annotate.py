"""Tests for tracker + annotate that don't need model weights."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.annotate import annotate
from pipeline.tracker import FrameTracker


def test_tracker_unknown_name():
    try:
        FrameTracker("nosuch", 30)
        raise AssertionError("should reject unknown tracker")
    except ValueError:
        pass


def test_tracker_real():
    """Real ultralytics ByteTracker with two synthetic moving-box frames."""
    t = FrameTracker("bytetrack", 30)
    det1 = np.array([[10.0, 10.0, 50.0, 50.0]])
    conf1 = np.array([0.9])
    cls1 = np.array([0])
    ids1 = t.update(0, det1, conf1, cls1)
    assert len(ids1) == 1 and ids1[0] is not None, f"expected ID, got {ids1}"
    # moving slightly -> same ID
    det2 = np.array([[12.0, 11.0, 52.0, 51.0]])
    ids2 = t.update(1, det2, conf1, cls1)
    assert len(ids2) == 1 and ids2[0] == ids1[0], f"expected same ID {ids1}, got {ids2}"


def test_tracker_no_dets():
    t = FrameTracker("bytetrack", 30)
    assert t.update(0, np.empty((0, 4)), np.empty((0,)), np.empty((0,))) == []


def test_annotate_shape_and_pixels():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    objs = [{"track_id": 1, "class_name": "person", "confidence": 0.9,
             "bbox_xyxy": [10.0, 10.0, 90.0, 90.0]}]
    out = annotate(frame.copy(), objs)
    assert out.shape == frame.shape
    assert out[10:90, 10, :].sum() > 0  # box border drawn
    # track_id None -> no crash, still drawn
    objs[0]["track_id"] = None
    annotate(frame.copy(), objs)


if __name__ == "__main__":
    test_tracker_unknown_name()
    test_tracker_no_dets()
    test_tracker_real()
    test_annotate_shape_and_pixels()
    print("all tracker/annotate tests passed")
