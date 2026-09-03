from __future__ import annotations

import os
import shutil
import subprocess

import pytest

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return bool(FFMPEG and FFPROBE and os.path.exists(FFMPEG))


requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not available")


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> str:
    """A small synthetic 640x360 clip with audio (8 s)."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    path = str(tmp_path_factory.mktemp("media") / "sample.mp4")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
           "-t", "8", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", path]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def software_env(monkeypatch, tmp_path):
    """Environment for running the service on the CPU inside tests."""
    monkeypatch.setenv("ENCODER_BACKEND", "software")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    if FFMPEG:
        monkeypatch.setenv("FFMPEG_BIN", FFMPEG)
    if FFPROBE:
        monkeypatch.setenv("FFPROBE_BIN", FFPROBE)
    for key in ("S3_BUCKET", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "RUNPOD_WEBHOOK_GET_JOB"):
        monkeypatch.delenv(key, raising=False)
    from video_compressor import capabilities

    capabilities.reset_cache()
    yield tmp_path
    capabilities.reset_cache()
