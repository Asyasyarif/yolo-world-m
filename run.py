"""Cog Runner — YOLO-World video + JSON timeline (PLAN §5).

Size dipilih saat build, bukan saat runtime: repo ini = "m".
(Repo yolo-timeline-video memakai "s" — dua model terpisah di Replicate
supaya kalkulasi harga per model bersih.)
"""

import os
import tempfile

from cog import BaseRunner, BaseModel, Input, Path

from pipeline.detector import Detector
from pipeline.describe import hex_to_hsv
from pipeline.image import process_image
from pipeline.process import process_video

MODEL_SIZE = os.environ.get("YOLO_WORLD_SIZE", "m")


class Output(BaseModel):
    video: Path
    timeline: Path


class Runner(BaseRunner):
    def setup(self):
        # satu model saja — ukuran fix saat build/di environment variable
        self.model = Detector(MODEL_SIZE)

    def run(
        self,
        video: Path = Input(description="Video input (atau gambar JPG/PNG)"),
        classes: str = Input(
            default="person,car,motorcycle,bicycle",
            description="Object classes, separated by commas",
        ),
        confidence: float = Input(default=0.25, ge=0.05, le=0.95),
        iou_threshold: float = Input(default=0.7, ge=0.05, le=0.95),
        tracker: str = Input(default="bytetrack", choices=["bytetrack", "botsort"]),
        show_confidence: bool = Input(default=True),
        show_track_id: bool = Input(default=True),
        preserve_audio: bool = Input(default=True),
        max_num_boxes: int = Input(
            default=10,
            ge=1, le=300,
            description="Max detections kept per frame (highest-confidence wins)",
        ),
        display_names: str = Input(
            default="",
            description=(
                "Optional display labels, comma-separated, same order as classes. "
                "Empty = use classes as-is. Detection uses the precise classes "
                "(e.g. 'soccer ball' for accuracy) while video/JSON show these "
                "(e.g. 'ball')."
            ),
        ),
        color_filter: str = Input(
            default="",
            description=(
                "Keep only detections whose pixel color matches, comma-separated. "
                "Options: red, orange, yellow, green, cyan, blue, purple, "
                "white, gray, black. Empty = keep all colors. "
                "Filters detections only; class_name is not renamed."
            ),
        ),
    ) -> Output:
        labels, colors = self._parse_classes(classes)

        disp = [x.strip() for x in display_names.split(",") if x.strip()]
        if disp and len(disp) != len(labels):
            raise ValueError(
                f"display_names must have same count as classes ({len(labels)}), got {len(disp)}"
            )
        display_labels = disp if disp else labels

        if not colors:  # explicit color_filter input still wins
            colors = [c.strip().lower() for c in color_filter.split(",") if c.strip()]

        # ponytail: no context manager — output dir must outlive Cog's file pickup
        workdir = tempfile.mkdtemp(prefix="video-timeline-")

        model = self.model
        is_image = str(video).lower().endswith((".jpg", ".jpeg", ".png"))
        if is_image:
            image_path, json_path = process_image(
                input_image=str(video),
                workdir=workdir,
                model=model,
                classes=labels,
                display_labels=display_labels,
                confidence=confidence,
                tracker=tracker,
                max_num_boxes=max_num_boxes,
                color_filter=colors,
            )
            return Output(video=Path(image_path), timeline=Path(json_path))

        mp4_path, json_path = process_video(
            input_video=str(video),
            workdir=workdir,
            model=model,
            classes=labels,
            display_labels=display_labels,
            confidence=confidence,
            tracker=tracker,
            preserve_audio=preserve_audio,
            max_num_boxes=max_num_boxes,
            color_filter=colors,
        )
        return Output(video=Path(mp4_path), timeline=Path(json_path))

    # ponytail: staticmethod supaya bisa dites tanpa instance Cog
    @staticmethod
    def _parse_classes(classes: str):
        """Prompt grammar: 'name @color-spec' per comma item.

        YOLO-World can't match color via prompt (measured: 'red soccer ball'
        scores 0.57 on a WHITE ball — hallucination). So color moves to a
        pixel filter, specified as hex/rgb/hsl after '@':
          'soccer ball @#808080'  -> detect 'soccer ball', keep gray pixels
          'ball @rgb(255,0,0)'    -> detect 'ball', keep red pixels
          'person'                -> detect 'person', no filter
        No color vocabulary, nothing hardcoded — any target color expressible.
        """
        import re

        items = [s.strip() for s in classes.split(",") if s.strip()]
        targets: list = []
        parsed = []
        for item in items:
            m = re.search(r"@(.+)$", item)
            name = re.sub(r"@(.+)$", "", item).strip()
            if name:
                parsed.append(name)
            if m:
                targets.append(hex_to_hsv(m.group(1).strip()))
            elif name and item != name:
                pass  # trailing @ alone
            if not m and item.endswith("@"):
                raise ValueError(f"color spec missing after '@' in '{item}'")
        if not parsed:
            raise ValueError("classes must not be empty")
        # align: target list is per-class (None where no color spec)
        # rebuild in order: for each parsed class, its target or None
        ordered_targets: list = []
        ti = 0
        for item in items:
            name = re.sub(r"@(.+)$", "", item).strip()
            if not name:
                continue
            if re.search(r"@(.+)$", item):
                ordered_targets.append(targets[ti])
                ti += 1
            else:
                ordered_targets.append(None)
        return parsed, ordered_targets
