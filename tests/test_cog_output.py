"""Keep Cog's declared file-list schema aligned with its actual output."""

import os
import sys
import tempfile
import zipfile
from pathlib import Path as LocalPath
from typing import get_type_hints

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cog import Path
from run import Runner


def test_output_contract():
    # A string/URI schema with an array response falls back to JSON in Replicate.
    assert get_type_hints(Runner.run)["return"] == list[Path], "Declare an array of file URIs"
    with tempfile.TemporaryDirectory() as directory:
        media = LocalPath(directory) / "output.mp4"
        timeline = LocalPath(directory) / "timeline.json"
        media.write_bytes(b"media")
        timeline.write_text("{}")
        for zipped in (False, True):
            output = Runner._package(media, timeline, zipped, directory, media.name, timeline.name)
            assert isinstance(output, list) and len(output) == 1
            assert isinstance(output[0], Path)
            if zipped:
                with zipfile.ZipFile(output[0]) as archive:
                    assert archive.namelist() == [media.name, timeline.name]
            else:
                assert output[0] == media


if __name__ == "__main__":
    test_output_contract()
    print("Cog output contract passed")
