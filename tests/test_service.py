"""CPU integration tests: run the real handler against a synthetic clip."""

import os

from conftest import requires_ffmpeg
from video_compressor.service import process_job


def _job(payload, job_id="test-job"):
    return {"id": job_id, "input": payload}


@requires_ffmpeg
def test_compress_local_success(software_env, sample_video):
    out_dir = software_env / "out"
    progress = []
    result = process_job(_job({
        "input_url": sample_video,
        "video": {"codec": "h264", "cq": 26, "max_height": 240},
        "output": {"type": "local", "path": str(out_dir)},
    }), progress_hook=progress.append)
    assert result["status"] == "success", result
    out = result["outputs"][0]
    assert out["pipeline"] == "software" and out["encoder"] == "libx264"
    assert (out["width"], out["height"]) == (426, 240) and abs(out["duration_seconds"] - 8.0) < 0.3
    assert out["audio_codec"] == "aac" and out["size_bytes"] > 0
    assert os.path.isfile(out["destination"]["path"])
    assert any(p.get("stage") == "encoding" for p in progress) and any(p.get("stage") == "uploading" for p in progress)
    assert result["input"]["width"] == 640 and "total_seconds" in result["timings"]
    assert not os.listdir(software_env / "work"), "workspace should be cleaned up"


@requires_ffmpeg
def test_trim_and_clips(software_env, sample_video):
    out_dir = software_env / "clips"
    result = process_job(_job({
        "input_url": sample_video,
        "video": {"cq": 30},
        "aspect": {"ratio": "9:16", "mode": "crop"},
        "clips": [
            {"name": "a", "trim": {"start": 1, "duration": 2}},
            {"name": "b", "trim": {"start": "00:00:05", "end": "00:00:07.5"}, "aspect": {"ratio": "1:1", "mode": "pad"}},
        ],
        "output": {"type": "local", "path": str(out_dir)},
    }))
    assert result["status"] == "success", result
    a, b = result["outputs"]
    assert abs(a["duration_seconds"] - 2.0) < 0.3 and (a["width"], a["height"]) == (202, 360)
    assert abs(b["duration_seconds"] - 2.5) < 0.3 and b["width"] == b["height"] == 640
    assert a["filename"].endswith("_a.mp4") and os.path.isfile(b["destination"]["path"])


@requires_ffmpeg
def test_trim_clamped_with_warning(software_env, sample_video):
    result = process_job(_job({
        "input_url": sample_video, "trim": {"start": 6, "end": 60}, "video": {"cq": 30},
        "output": {"type": "local", "path": str(software_env / "t.mp4")},
    }))
    assert result["status"] == "success", result
    assert abs(result["outputs"][0]["duration_seconds"] - 2.0) < 0.3
    assert any("clamped" in w for w in result["outputs"][0]["warnings"])


@requires_ffmpeg
def test_base64_output(software_env, sample_video):
    result = process_job(_job({
        "input_url": sample_video, "trim": {"duration": 1}, "video": {"cq": 35, "max_height": 120},
        "audio": {"codec": "none"}, "output": {"type": "base64"},
    }))
    assert result["status"] == "success", result
    assert result["outputs"][0]["destination"]["data_base64"].startswith("AAAA")


def test_invalid_payload_is_structured_error(software_env):
    result = process_job(_job({"input_url": "https://x/y.mp4", "video": {"codec": "vp9"}}))
    assert result["status"] == "error" and result["error"].startswith("INVALID_INPUT")
    assert result["error_detail"]["code"] == "INVALID_INPUT" and result["error_detail"]["details"]["problems"]
    result = process_job({"id": "x"})
    assert result["error_detail"]["code"] == "INVALID_INPUT"


def test_missing_local_input(software_env):
    result = process_job(_job({"input_url": "/nope/missing.mp4", "output": {"type": "base64"}}))
    assert result["error_detail"]["code"] == "INVALID_INPUT" and "does not exist" in result["error"]


def test_no_destination_is_explicit(software_env):
    result = process_job(_job({"input_url": "/nope/missing.mp4"}))
    assert result["error_detail"]["code"] == "STORAGE_NOT_CONFIGURED"


@requires_ffmpeg
def test_corrupt_input(software_env):
    bad = software_env / "bad.mp4"
    bad.write_bytes(b"not a video at all" * 100)
    result = process_job(_job({"input_url": str(bad), "output": {"type": "base64"}}))
    assert result["status"] == "error" and result["error_detail"]["code"] in ("PROBE_FAILED", "UNSUPPORTED_MEDIA")
    assert result["error_detail"].get("hint")


@requires_ffmpeg
def test_trim_start_beyond_duration(software_env, sample_video):
    result = process_job(_job({"input_url": sample_video, "trim": {"start": 100}, "output": {"type": "base64"}}))
    assert result["error_detail"]["code"] == "INVALID_INPUT" and "beyond" in result["error"]


@requires_ffmpeg
def test_encode_timeout(software_env, sample_video):
    result = process_job(_job({
        "input_url": sample_video, "video": {"preset": "p7", "cq": 10, "width": 1920, "height": 1080},
        "output": {"type": "base64"}, "options": {"timeout_seconds": 1},
    }))
    assert result["status"] == "error", result
    assert result["error_detail"]["code"] == "ENCODE_TIMEOUT" and "hint" in result["error_detail"]


@requires_ffmpeg
def test_strict_nvenc_without_gpu_fails_clearly(software_env, sample_video, monkeypatch):
    monkeypatch.setenv("ENCODER_BACKEND", "nvenc")
    from video_compressor import capabilities

    capabilities.reset_cache()
    result = process_job(_job({"input_url": sample_video, "output": {"type": "base64"}}))
    assert result["status"] == "error"
    assert result["error_detail"]["code"] in ("GPU_LIBRARIES_NOT_VISIBLE", "NVENC_UNAVAILABLE", "GPU_ERROR")
    assert result.get("refresh_worker") is True and result["error_detail"]["hint"]
