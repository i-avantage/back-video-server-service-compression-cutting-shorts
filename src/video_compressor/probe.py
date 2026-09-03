"""ffprobe wrapper returning a normalised :class:`MediaInfo`."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .errors import ProbeError, UnsupportedMediaError

log = logging.getLogger(__name__)


@dataclass
class VideoStreamInfo:
    codec: str
    width: int
    height: int
    pix_fmt: str | None
    fps: float | None
    rotation: int = 0
    bit_rate: int | None = None
    profile: str | None = None
    nb_frames: int | None = None

    @property
    def display_width(self) -> int:
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation in (90, 270) else self.height


@dataclass
class AudioStreamInfo:
    codec: str
    channels: int | None
    sample_rate: int | None
    bit_rate: int | None = None


@dataclass
class MediaInfo:
    path: str
    format_name: str
    duration: float
    size_bytes: int
    bit_rate: int | None
    video: VideoStreamInfo | None
    audio: AudioStreamInfo | None
    stream_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "container": self.format_name,
            "duration_seconds": round(self.duration, 3),
            "size_bytes": self.size_bytes,
            "bit_rate": self.bit_rate,
        }
        if self.video:
            out.update(
                {
                    "video_codec": self.video.codec,
                    "width": self.video.display_width,
                    "height": self.video.display_height,
                    "pixel_format": self.video.pix_fmt,
                    "fps": round(self.video.fps, 3) if self.video.fps else None,
                    "rotation": self.video.rotation,
                }
            )
        out["audio_codec"] = self.audio.codec if self.audio else None
        if self.audio:
            out["audio_channels"] = self.audio.channels
            out["audio_sample_rate"] = self.audio.sample_rate
        return out


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _parse_fps(stream: dict[str, Any]) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key)
        if not raw or raw in ("0/0", "N/A"):
            continue
        try:
            value = float(Fraction(raw))
        except (ValueError, ZeroDivisionError):
            continue
        if 0 < value < 1000:
            return value
    return None


def _parse_rotation(stream: dict[str, Any]) -> int:
    rotation: float | None = None
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = _to_float(side.get("rotation"))
            break
    if rotation is None:
        rotation = _to_float((stream.get("tags") or {}).get("rotate"))
    if rotation is None:
        return 0
    return int(round(rotation)) % 360


def run_ffprobe(ffprobe_bin: str, path: str, *, timeout: int = 120) -> dict[str, Any]:
    cmd = [
        ffprobe_bin, "-hide_banner", "-v", "error",
        "-print_format", "json", "-show_format", "-show_streams", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise ProbeError(
            f"ffprobe binary not found: {ffprobe_bin!r}.",
            code="FFPROBE_NOT_FOUND",
            hint="Install FFmpeg (with ffprobe) or set FFPROBE_BIN.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out after {timeout}s on {os.path.basename(path)}.") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise ProbeError(
            f"ffprobe failed (exit {proc.returncode}) on {os.path.basename(path)}: {stderr[-500:] or 'no stderr'}",
            details={"stderr": stderr[-2000:], "returncode": proc.returncode},
            hint="The file is probably not a valid/complete video, or the download returned an error page.",
        )
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON.", details={"stdout": proc.stdout[-1000:]}) from exc


def probe_media(ffprobe_bin: str, path: str, *, timeout: int = 120) -> MediaInfo:
    """Probe ``path`` and return a :class:`MediaInfo`. Requires a video stream."""
    data = run_ffprobe(ffprobe_bin, path, timeout=timeout)
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"
                         and s.get("disposition", {}).get("attached_pic", 0) != 1), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise UnsupportedMediaError(
            "The input has no video stream.",
            details={"format": fmt.get("format_name"), "streams": [s.get("codec_type") for s in streams]},
        )

    width = _to_int(video_stream.get("width")) or 0
    height = _to_int(video_stream.get("height")) or 0
    if width <= 0 or height <= 0:
        raise UnsupportedMediaError(
            "Could not determine the input video resolution.", details={"stream": video_stream}
        )

    duration = _to_float(fmt.get("duration")) or _to_float(video_stream.get("duration"))
    if duration is None:
        # Some containers (raw streams) do not report a duration; estimate from frames.
        nb_frames = _to_int(video_stream.get("nb_frames"))
        fps = _parse_fps(video_stream)
        if nb_frames and fps:
            duration = nb_frames / fps
    if duration is None or duration <= 0:
        raise UnsupportedMediaError(
            "Could not determine the input duration (unsupported or truncated file).",
            details={"format": fmt.get("format_name")},
        )

    video = VideoStreamInfo(
        codec=str(video_stream.get("codec_name") or "unknown"),
        width=width,
        height=height,
        pix_fmt=video_stream.get("pix_fmt"),
        fps=_parse_fps(video_stream),
        rotation=_parse_rotation(video_stream),
        bit_rate=_to_int(video_stream.get("bit_rate")),
        profile=video_stream.get("profile"),
        nb_frames=_to_int(video_stream.get("nb_frames")),
    )
    audio = None
    if audio_stream is not None:
        audio = AudioStreamInfo(
            codec=str(audio_stream.get("codec_name") or "unknown"),
            channels=_to_int(audio_stream.get("channels")),
            sample_rate=_to_int(audio_stream.get("sample_rate")),
            bit_rate=_to_int(audio_stream.get("bit_rate")),
        )
    size = _to_int(fmt.get("size"))
    if size is None:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
    return MediaInfo(
        path=path,
        format_name=str(fmt.get("format_name") or "unknown"),
        duration=duration,
        size_bytes=size,
        bit_rate=_to_int(fmt.get("bit_rate")),
        video=video,
        audio=audio,
        stream_count=len(streams),
        raw=data,
    )
