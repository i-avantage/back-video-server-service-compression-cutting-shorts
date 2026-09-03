"""Parsing/formatting helpers for durations, bitrates and sizes."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

Number = int | float

_BITRATE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmMgG]?)(?:b(?:ps|it/?s)?)?\s*$")
_TIMECODE_RE = re.compile(r"^\s*(?:(\d+):)?(?:(\d{1,2}):)?(\d{1,2}(?:\.\d+)?)\s*$")


def parse_duration(value: str | Number, *, field: str = "time") -> float:
    """Parse ``12.5``, ``"12.5"``, ``"01:23"`` or ``"00:01:23.500"`` to seconds."""
    if isinstance(value, bool):
        raise ValueError(f"{field}: booleans are not valid times")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field}: empty value")
        try:
            seconds = float(text)
        except ValueError:
            match = _TIMECODE_RE.match(text)
            if not match:
                raise ValueError(
                    f"{field}: {value!r} is not a valid time (use seconds, MM:SS or HH:MM:SS[.mmm])"
                ) from None
            hours, minutes, secs = match.groups()
            if minutes is None:  # "MM:SS" -> groups are (MM, None, SS)
                minutes, hours = hours, None
            seconds = float(secs) + 60 * int(minutes or 0) + 3600 * int(hours or 0)
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN / inf
        raise ValueError(f"{field}: {value!r} is not a finite time")
    if seconds < 0:
        raise ValueError(f"{field}: must not be negative (got {seconds})")
    return seconds


def parse_bitrate(value: str | Number, *, field: str = "bitrate") -> int:
    """Parse ``"5M"``, ``"800k"``, ``5000000`` to bits per second (int)."""
    if isinstance(value, bool):
        raise ValueError(f"{field}: booleans are not valid bitrates")
    if isinstance(value, (int, float)):
        bits = float(value)
    else:
        match = _BITRATE_RE.match(str(value))
        if not match:
            raise ValueError(
                f"{field}: {value!r} is not a valid bitrate (examples: '5M', '800k', 5000000)"
            )
        number, unit = match.groups()
        multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000}[unit.lower()]
        bits = float(number) * multiplier
    if bits <= 0:
        raise ValueError(f"{field}: must be positive (got {value!r})")
    if bits > 2_000_000_000:
        raise ValueError(f"{field}: {value!r} is unreasonably high (max 2 Gbit/s)")
    return int(round(bits))


def format_seconds(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` for FFmpeg arguments and logs."""
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1000 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1000
    return f"{num:.1f} TB"  # pragma: no cover


def redact_url(url: str) -> str:
    """Strip query string / credentials from URLs before logging or returning them.

    Signed URLs carry secrets in the query string; never echo them back.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "<credentials>@" + netloc.rsplit("@", 1)[1]
    query = "<redacted>" if parts.query else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def even(value: int) -> int:
    """Round down to an even number (codec requirement for 4:2:0 chroma)."""
    return value - (value % 2)
