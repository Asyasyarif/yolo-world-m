"""ByteTrack/BoT-SORT tracking.

Directly instantiates Ultralytics' trackers per prediction (model.track()
keeps state on the model instance, which would leak across predictions —
PLAN §5.3). Ultralytics trackers expect a Results-like object exposing
`conf`, `cls`, `xywh` and boolean indexing; their update() output carries
the source detection index per row, so we map track IDs exactly.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

TRACKER_DEFAULTS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


class _Dets:
    """Minimal detection container the ultralytics trackers understand."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack(
            [(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2, w, h],
            axis=1,
        )

    def __getitem__(self, mask) -> "_Dets":
        return _Dets(self.xyxy[mask], self.conf[mask], self.cls[mask])

    def __len__(self) -> int:
        return len(self.conf)


class FrameTracker:
    """One instance per prediction. Call update() frame by frame, then discard."""

    def __init__(self, tracker: str, fps: float):
        if tracker not in TRACKER_DEFAULTS:
            raise ValueError(
                f"unknown tracker '{tracker}', choose from {sorted(TRACKER_DEFAULTS)}"
            )
        self.tracker = self._make(tracker, fps)

    @staticmethod
    def _make(name: str, fps: float):
        import yaml
        from ultralytics.utils import IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        from ultralytics.trackers import BOTSORT, BYTETracker

        cfg = check_yaml(TRACKER_DEFAULTS[name])
        args = IterableSimpleNamespace(**yaml.safe_load(open(cfg)))
        args.track_buffer = max(1, int(round(fps * 0.5)))  # ~0.5s occlusion
        if name == "bytetrack":
            return BYTETracker(args)
        return BOTSORT(args)

    def update(
        self, frame_idx: int, dets: np.ndarray, confs: np.ndarray, clss: np.ndarray
    ) -> List[Optional[int]]:
        """Feed one frame's detections; return track_id per detection (None if
        the tracker dropped it, e.g. below new-track threshold)."""
        if len(dets) == 0:
            return []
        packed = _Dets(dets, confs, clss)
        tracks = np.asarray(self.tracker.update(packed))
        if tracks.size == 0:
            return [None] * len(dets)
        # rows: [x1, y1, x2, y2, track_id, score, cls, idx]
        out: List[Optional[int]] = [None] * len(dets)
        for row in tracks.reshape(len(tracks), -1):
            det_idx = int(row[7])
            if 0 <= det_idx < len(out):
                out[det_idx] = int(row[4])
        return out


__all__ = ["FrameTracker", "TRACKER_DEFAULTS"]
