"""End-to-end test that runs the full process_video pipeline without YOLO weights,
by swapping in a fake Detector that returns fixed boxes."""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.process import process_video


class FakeDetector:
    """Pretends to be a Detector: returns a stationary box every frame."""

    model_size = "s"

    def set_classes(self, labels):
        self.labels = labels

    def detect(self, img, conf=0.25, iou=0.7):
        class _Boxes:
            def __init__(self):
                self.xyxy = np.array([[10.0, 10.0, 50.0, 50.0]])
                self.conf = np.array([0.92])
                self.cls = np.array([0])
            def __len__(self):
                return 1
        class _Res:
            boxes = _Boxes()
        return [_Res()]


def _make_test_video(path, with_audio=False):
    size = "128x72"
    cmd = ["ffmpeg", "-nostdin", "-y",
           "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration=2"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += [path]
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_frames(path):
    import av
    n = 0
    with av.open(path) as c:
        for _ in c.decode(video=0):
            n += 1
    return n


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.mp4")
        _make_test_video(src, with_audio=True)
        work = os.path.join(d, "work")
        mp4, js = process_video(
            input_video=src, workdir=work, model=FakeDetector(),
            classes=["person"], confidence=0.25, tracker="bytetrack",
            preserve_audio=True,
        )
        assert os.path.exists(mp4) and os.path.exists(js)

        # acceptance: len(frames) == decoded frame count of output
        with open(js) as f:
            doc = json.load(f)
        n = _probe_frames(mp4)
        assert len(doc["frames"]) == n, f"{len(doc['frames'])} frames vs {n} decoded"
        assert doc["video"]["total_frames"] == n

        # acceptance: frame_index monotonic 0..n-1, no gaps, no dupes
        idx = [fr["frame_index"] for fr in doc["frames"]]
        assert idx == list(range(n))

        # acceptance: timestamps monotonically increasing
        ts = [fr["timestamp_ms"] for fr in doc["frames"]]
        assert all(b > a for a, b in zip(ts, ts[1:]))

        # acceptance: every bbox valid, every frame's objects in-bounds and confident
        w, h = doc["video"]["width"], doc["video"]["height"]
        for fr in doc["frames"]:
            for o in fr["objects"]:
                x1, y1, x2, y2 = o["bbox_xyxy"]
                assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
                assert 0.0 <= o["confidence"] <= 1.0
                assert o["class_name"] == "person"
                assert o["track_id"] is not None  # fake box never moves

        # acceptance: schema version present
        assert doc["schema_version"] == "1.0.0"

        # audio preserved end to end
        from pipeline.video_io import probe
        info = probe(mp4)
        assert info.has_audio, "audio input should be preserved in output"


def test_invalid_video():
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.mp4")
        open(bad, "wb").write(b"nonsense")
        try:
            process_video(bad, d, FakeDetector(), ["person"])
            raise AssertionError("should reject")
        except Exception as e:
            assert "video" in str(e).lower() or "input" in str(e).lower()


if __name__ == "__main__":
    test_invalid_video()
    test_end_to_end()
    print("all end-to-end tests passed")
