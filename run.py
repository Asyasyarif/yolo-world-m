"""Cog Runner — YOLO-World video + timeline (PLAN §5).

Size dipilih saat build, bukan saat runtime: repo ini = "m".
(Repo yolo-timeline-video memakai "s" — dua model terpisah di Replicate
supaya kalkulasi harga per model bersih.)

Output berupa list berisi satu file, sesuai schema array URI di Replicate:
  output_zip=True  -> ZIP (output.mp4 + timeline.json)
  output_zip=False -> MP4/JPG untuk preview
"""

import os
import tempfile
import zipfile

from cog import BaseRunner, Input, Path

from pipeline.detector import Detector
from pipeline.describe import hex_to_hsv
from pipeline.image import process_image
from pipeline.process import process_video

MODEL_SIZE = os.environ.get("YOLO_WORLD_SIZE", "m")


class Runner(BaseRunner):
    def setup(self):
        # satu model saja — ukuran fix saat build/di environment variable
        self.model = Detector(MODEL_SIZE)

    def run(
        self,
        video: Path = Input(description="Input video (MP4/MOV) or image (JPG/PNG)"),
        classes: str = Input(
            default="person,car,motorcycle,bicycle",
            description=(
                "Open-vocabulary class prompts, comma-separated. "
                "Optionally attach a pixel color filter per class with @hex: "
                "e.g. 'soccer ball @#808080' keeps only gray-ish detections. "
                "Use English words — the text encoder is English-only."
            ),
        ),
        confidence: float = Input(default=0.25, ge=0.05, le=0.95,
                                  description="Detection confidence threshold"),
        iou_threshold: float = Input(default=0.7, ge=0.05, le=0.95,
                                     description="NMS IoU threshold"),
        tracker: str = Input(default="bytetrack", choices=["bytetrack", "botsort"],
                             description="Multi-object tracker"),
        show_confidence: bool = Input(default=True,
                                      description="Show confidence % in video labels"),
        show_track_id: bool = Input(default=True,
                                    description="Show track ID in video labels"),
        preserve_audio: bool = Input(default=True,
                                     description="Copy the input's audio track to the output MP4"),
        max_num_boxes: int = Input(
            default=10,
            ge=1, le=300,
            description="Max detections kept per frame (highest-confidence wins)",
        ),
        output_zip: bool = Input(
            default=False,
            description=(
                "Deliver the result as a single ZIP (video + timeline.json). "
                "Unchecked = the annotated video file itself, shown as a "
                "playable preview in the UI."
            ),
        ),
    ) -> list[Path]:
        labels, colors = self._parse_classes(classes)

        display_labels = labels

        # ponytail: no context manager — output dir must outlive Cog's file pickup
        workdir = tempfile.mkdtemp(prefix="video-timeline-")

        model = self.model
        is_image = str(video).lower().endswith((".jpg", ".jpeg", ".png"))
        if is_image:
            image_path, json_path, _tl = process_image(
                input_image=str(video),
                workdir=workdir,
                model=model,
                classes=labels,
                display_labels=display_labels,
                confidence=confidence,
                iou_threshold=iou_threshold,
                tracker=tracker,
                max_num_boxes=max_num_boxes,
                color_filter=colors,
            )
            return self._package(
                image_path, json_path, output_zip, workdir, "output.jpg", "timeline.json"
            )

        mp4_path, json_path, timeline_obj = process_video(
            input_video=str(video),
            workdir=workdir,
            model=model,
            classes=labels,
            display_labels=display_labels,
            confidence=confidence,
            iou_threshold=iou_threshold,
            tracker=tracker,
            preserve_audio=preserve_audio,
            max_num_boxes=max_num_boxes,
            color_filter=colors,
        )
        return self._package(
            mp4_path, json_path, output_zip, workdir, "output.mp4", "timeline.json"
        )

    @staticmethod
    def _package(primary_path, timeline_path, wants_zip, workdir, out_name, tl_name) -> list[Path]:
        """One file in a list: media preview or a downloadable ZIP bundle."""
        # ponytail: explicit file-list schema matches the array response on Replicate.
        if not wants_zip:
            return [Path(primary_path)]
        zip_path = os.path.join(workdir, "output.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(primary_path, arcname=out_name)
            zf.write(timeline_path, arcname=tl_name)
        return [Path(zip_path)]

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
