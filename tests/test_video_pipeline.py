"""Smoke tests (no GPU, no model) for video_io + timeline."""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.timeline import FrameWriter, make_video_meta
from pipeline.video_io import VideoError, probe, validate_video, Writer


def _make_test_video(path, with_audio=False, fps=30, seconds=1):
    size = "128x72"
    vf = f"ffmpeg -nostdin -y -f lavfi -i testsrc=size={size}:rate={fps}:duration={seconds}"
    if with_audio:
        vf += " -f lavfi -i sine=frequency=440:duration=%s" % seconds
        vf += " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest"
    else:
        vf += " -c:v libx264 -pix_fmt yuv420p"
    subprocess.run(vf.split() + [path], check=True, capture_output=True)


def test_probe_and_validation():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.mp4")
        _make_test_video(p)
        info = probe(p)
        assert info.has_video and info.width == 128 and info.height == 72
        assert info.fps == 30 and info.duration_s >= 1
        validate_video(info)  # should not raise

        # over-limit video is rejected
        with tempfile.TemporaryDirectory() as d2:
            p2 = os.path.join(d2, "long.mp4")
            _make_test_video(p2, seconds=2)
            import unittest.mock as mock
            with mock.patch("pipeline.video_io.MAX_DURATION_S", 1.0):
                try:
                    validate_video(probe(p2))
                    raise AssertionError("should reject over-limit")
                except VideoError:
                    pass

        # non-video file rejected
        bad = os.path.join(d, "bad.mp4")
        open(bad, "wb").write(b"this is not a video")
        try:
            probe(bad)
            raise AssertionError("should reject non-video")
        except VideoError:
            pass


def test_writer_and_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.mp4")
        _make_test_video(p)
        info = probe(p)
        out = os.path.join(d, "out.mp4")
        import av
        w = Writer(out, info.width, info.height, 30)
        count = 0
        with av.open(p) as c:
            for f in c.decode(video=0):
                w.write(f.to_ndarray(format="bgr24"))
                count += 1
        w.close()
        assert w.count == count
        assert probe(out).fps == 30  # valid, decodable output


def test_audio_pipeline():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.mp4")
        _make_test_video(p, with_audio=True)
        assert probe(p).has_audio
        a = os.path.join(d, "audio.m4a")
        m = os.path.join(d, "muxed.mp4")
        from pipeline.video_io import extract_audio, mux_audio
        extract_audio(p, a)
        assert os.path.exists(a) and os.path.getsize(a) > 0
        mux_audio(p, a, m)
        assert probe(m).has_audio


def test_timeline_frame_writer():
    with tempfile.TemporaryDirectory() as d:
        j = os.path.join(d, "timeline.json")
        jl = os.path.join(d, ".frames.jsonl")
        fw = FrameWriter(jl)
        for i in range(3):
            fw.add({"frame_index": i, "timestamp_ms": i * 33.333,
                    "source_timestamp_ms": i * 33.333, "objects": []})
        fw.close()
        meta = make_video_meta("output.mp4", 128, 72, 30, 3, 100)
        header = {"schema_version": "1.0.0", "video": meta,
                  "model": {"tracker": "bytetrack"}}
        doc = fw.finalize(j, header)
        assert doc["schema_version"] == "1.0.0"
        assert doc["video"]["fps_num"] == 30
        assert [fr["frame_index"] for fr in doc["frames"]] == [0, 1, 2]
        assert not os.path.exists(jl)  # temp cleaned


if __name__ == "__main__":
    test_probe_and_validation()
    test_writer_and_roundtrip()
    test_audio_pipeline()
    test_timeline_frame_writer()
    print("all video_io/timeline tests passed")
