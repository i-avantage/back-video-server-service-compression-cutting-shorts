import pytest

from video_compressor.units import format_seconds, parse_bitrate, parse_duration, redact_url


@pytest.mark.parametrize("value,expected", [
    (5, 5.0), (2.5, 2.5), ("7", 7.0), ("1:30", 90.0), ("01:02:03", 3723.0), ("00:00:03.250", 3.25), ("90.5", 90.5),
])
def test_parse_duration(value, expected):
    assert parse_duration(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "-1", -3, "1:2:3:4", "nan", True])
def test_parse_duration_rejects(value):
    with pytest.raises(ValueError):
        parse_duration(value)


@pytest.mark.parametrize("value,expected", [
    ("5M", 5_000_000), ("800k", 800_000), ("128K", 128_000), (5000000, 5_000_000), ("1.5m", 1_500_000), ("2Mbps", 2_000_000),
])
def test_parse_bitrate(value, expected):
    assert parse_bitrate(value) == expected


@pytest.mark.parametrize("value", ["", "fast", "0", -5, "5T"])
def test_parse_bitrate_rejects(value):
    with pytest.raises(ValueError):
        parse_bitrate(value)


def test_format_seconds():
    assert format_seconds(3723.25) == "01:02:03.250"
    assert format_seconds(0) == "00:00:00.000"


def test_redact_url():
    assert redact_url("https://b.s3.amazonaws.com/k.mp4?X-Amz-Signature=abc") == "https://b.s3.amazonaws.com/k.mp4?<redacted>"
    assert redact_url("https://user:pw@host/x.mp4") == "https://<credentials>@host/x.mp4"
    assert redact_url("https://host/x.mp4") == "https://host/x.mp4"
