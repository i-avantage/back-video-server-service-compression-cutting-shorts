import pytest

from video_compressor.errors import InputValidationError
from video_compressor.schema import parse_job_input


def test_defaults():
    job = parse_job_input({"input_url": "https://x/y.mp4"})
    assert job.video.codec == "h264" and job.video.rate_control == "cq" and job.video.cq == 23
    assert job.audio.codec == "aac" and job.audio.bitrate_bps == 128_000
    assert job.container == "mp4" and job.options.hw_decode == "auto"


def test_url_aliases():
    assert parse_job_input({"url": "https://x/y.mp4"}).input_url == "https://x/y.mp4"
    assert parse_job_input({"video_url": "s3://b/k.mp4"}).input_url == "s3://b/k.mp4"


def test_missing_input():
    with pytest.raises(InputValidationError) as exc:
        parse_job_input(None)
    assert exc.value.code == "INVALID_INPUT"
    with pytest.raises(InputValidationError):
        parse_job_input({"video": {}})


def test_unknown_field_rejected():
    with pytest.raises(InputValidationError) as exc:
        parse_job_input({"input_url": "https://x/y.mp4", "video": {"bitrat": "5M"}})
    assert "bitrat" in exc.value.message
    assert exc.value.details["problems"][0]["field"] == "video.bitrat"


def test_codec_aliases_and_invalid():
    assert parse_job_input({"input_url": "https://x/y.mp4", "video": {"codec": "H.265"}}).video.codec == "hevc"
    with pytest.raises(InputValidationError):
        parse_job_input({"input_url": "https://x/y.mp4", "video": {"codec": "vp9"}})


def test_rate_control_rules():
    with pytest.raises(InputValidationError, match="bitrate is required"):
        parse_job_input({"input_url": "https://x/y.mp4", "video": {"rate_control": "vbr"}})
    with pytest.raises(InputValidationError, match="ignored in 'cq' mode"):
        parse_job_input({"input_url": "https://x/y.mp4", "video": {"bitrate": "5M"}})
    job = parse_job_input({"input_url": "https://x/y.mp4", "video": {"rate_control": "vbr", "bitrate": "5M", "max_bitrate": "8M"}})
    assert job.video.bitrate_bps == 5_000_000 and job.video.max_bitrate_bps == 8_000_000


def test_resolution_rules():
    with pytest.raises(InputValidationError, match="even"):
        parse_job_input({"input_url": "https://x/y.mp4", "video": {"width": 1281}})
    with pytest.raises(InputValidationError, match="either width/height or max"):
        parse_job_input({"input_url": "https://x/y.mp4", "video": {"width": 1280, "max_height": 720}})


def test_trim_parsing():
    job = parse_job_input({"input_url": "https://x/y.mp4", "trim": {"start": "00:01:00", "duration": 30}})
    assert job.trim.start_seconds == 60 and job.trim.end_seconds == 90
    with pytest.raises(InputValidationError, match="either 'end' or 'duration'"):
        parse_job_input({"input_url": "https://x/y.mp4", "trim": {"end": 10, "duration": 5}})
    with pytest.raises(InputValidationError, match="greater than start"):
        parse_job_input({"input_url": "https://x/y.mp4", "trim": {"start": 10, "end": 5}})


def test_clips_rules():
    base = {"input_url": "https://x/y.mp4"}
    with pytest.raises(InputValidationError, match="not both"):
        parse_job_input({**base, "trim": {"start": 1}, "clips": [{"name": "a", "trim": {"start": 1}}]})
    with pytest.raises(InputValidationError, match="unique"):
        parse_job_input({**base, "clips": [{"name": "a", "trim": {"start": 1}}, {"name": "a", "trim": {"start": 2}}]})
    with pytest.raises(InputValidationError, match="no spaces"):
        parse_job_input({**base, "clips": [{"name": "bad name", "trim": {"start": 1}}]})


def test_output_rules():
    base = {"input_url": "https://x/y.mp4"}
    with pytest.raises(InputValidationError, match="output.url"):
        parse_job_input({**base, "output": {"type": "presigned_put"}})
    with pytest.raises(InputValidationError, match="output.path"):
        parse_job_input({**base, "output": {"type": "local"}})
    with pytest.raises(InputValidationError, match="relative object key"):
        parse_job_input({**base, "output": {"type": "s3", "key": "../etc/x"}})
    out = parse_job_input({**base, "output": {"type": "s3", "key": "a/b.mp4", "overwrite": True}}).output
    assert out.key == "a/b.mp4" and out.overwrite


def test_container_codec_rules():
    with pytest.raises(InputValidationError, match="av1 cannot be stored"):
        parse_job_input({"input_url": "https://x/y.mp4", "container": "mov", "video": {"codec": "av1"}})
    with pytest.raises(InputValidationError):
        parse_job_input({"input_url": "https://x/y.mp4", "container": "webm"})
    with pytest.raises(InputValidationError, match="10-bit"):
        parse_job_input({"input_url": "https://x/y.mp4", "video": {"codec": "h264", "pixel_format": "p010le"}})


def test_aspect_rules():
    job = parse_job_input({"input_url": "https://x/y.mp4", "aspect": {"ratio": "9:16"}})
    assert (job.aspect.ratio_w, job.aspect.ratio_h) == (9, 16)
    with pytest.raises(InputValidationError, match="W:H"):
        parse_job_input({"input_url": "https://x/y.mp4", "aspect": {"ratio": "vertical"}})
