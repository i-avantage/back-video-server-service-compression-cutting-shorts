"""Job input schema.

Strict by design: unknown fields are rejected so a typo (``"bitrat"``) never
silently falls back to a default. All values are normalised here so the
rest of the code can trust them.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import InputValidationError
from .units import parse_bitrate, parse_duration

VideoCodec = Literal["h264", "hevc", "av1"]
Container = Literal["mp4", "mov", "mkv"]
Preset = Literal["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
PixelFormat = Literal["yuv420p", "nv12", "p010le", "yuv444p"]

_CODEC_ALIASES = {
    "h264": "h264", "avc": "h264", "x264": "h264", "h.264": "h264",
    "hevc": "hevc", "h265": "hevc", "x265": "hevc", "h.265": "hevc",
    "av1": "av1",
}
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RATIO_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")

CONTAINER_EXTENSION = {"mp4": "mp4", "mov": "mov", "mkv": "mkv"}
CONTAINER_CONTENT_TYPE = {"mp4": "video/mp4", "mov": "video/quicktime", "mkv": "video/x-matroska"}
#: Audio codecs that can be stream-copied into each container without remuxing issues.
CONTAINER_AUDIO_COPY_OK = {
    "mp4": {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"},
    "mov": {"aac", "mp3", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s24le", "pcm_s16be", "pcm_s24be"},
    "mkv": None,  # Matroska accepts practically everything.
}
CODEC_PROFILES = {
    "h264": {"baseline", "main", "high", "high444p"},
    "hevc": {"main", "main10", "rext"},
    "av1": {"main"},
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)


class TrimSpec(StrictModel):
    """Cut a segment. ``start`` defaults to 0; give ``end`` or ``duration`` (not both)."""

    start: str | float | int | None = None
    end: str | float | int | None = None
    duration: str | float | int | None = None

    start_seconds: float = Field(default=0.0, exclude=True)
    end_seconds: float | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _normalise(self) -> TrimSpec:
        start = parse_duration(self.start, field="trim.start") if self.start is not None else 0.0
        end: float | None = None
        if self.end is not None and self.duration is not None:
            raise ValueError("trim: give either 'end' or 'duration', not both")
        if self.end is not None:
            end = parse_duration(self.end, field="trim.end")
            if end <= start:
                raise ValueError(f"trim: end ({end}s) must be greater than start ({start}s)")
        elif self.duration is not None:
            duration = parse_duration(self.duration, field="trim.duration")
            if duration <= 0:
                raise ValueError("trim: duration must be greater than 0")
            end = start + duration
        if start == 0.0 and end is None:
            raise ValueError("trim: specify at least one of start, end or duration")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        return self

    @property
    def duration_seconds(self) -> float | None:
        return None if self.end_seconds is None else self.end_seconds - self.start_seconds


class AspectSpec(StrictModel):
    """Reframe to a target aspect ratio (e.g. 9:16 for shorts) by cropping or padding."""

    ratio: str = Field(description="Target aspect ratio, e.g. '9:16', '1:1', '4:5', '16:9'.")
    mode: Literal["crop", "pad"] = "crop"
    anchor: Literal["center", "top", "bottom", "left", "right"] = "center"
    pad_color: str = "black"

    ratio_w: int = Field(default=0, exclude=True)
    ratio_h: int = Field(default=0, exclude=True)

    @model_validator(mode="after")
    def _normalise(self) -> AspectSpec:
        match = _RATIO_RE.match(self.ratio)
        if not match:
            raise ValueError(f"aspect.ratio {self.ratio!r} must look like 'W:H', e.g. '9:16'")
        w, h = int(match.group(1)), int(match.group(2))
        if w <= 0 or h <= 0:
            raise ValueError("aspect.ratio components must be positive integers")
        if not re.match(r"^[A-Za-z0-9#@.]+$", self.pad_color):
            raise ValueError(f"aspect.pad_color {self.pad_color!r} is not a valid FFmpeg colour")
        object.__setattr__(self, "ratio_w", w)
        object.__setattr__(self, "ratio_h", h)
        return self


class VideoSpec(StrictModel):
    codec: str = Field(default="h264", description="h264 (default), hevc or av1.")
    preset: Preset = Field(default="p5", description="NVENC preset p1 (fastest) .. p7 (best quality).")
    tune: Literal["hq", "ll", "ull", "lossless"] = "hq"
    rate_control: Literal["cq", "vbr", "cbr"] = Field(
        default="cq", description="cq = constant quality (CRF-like, default), vbr/cbr need 'bitrate'."
    )
    cq: int = Field(default=23, ge=0, le=63, description="Quality for cq mode (lower = better, 18-28 typical).")
    bitrate: str | int | None = Field(default=None, description="Target bitrate for vbr/cbr, e.g. '5M'.")
    max_bitrate: str | int | None = None
    buffer_size: str | int | None = None
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    max_width: int | None = Field(default=None, ge=16, le=8192, description="Downscale to fit (never upscales).")
    max_height: int | None = Field(default=None, ge=16, le=8192)
    fps: float | None = Field(default=None, gt=0, le=240)
    pixel_format: PixelFormat | None = None
    profile: str | None = None
    level: str | None = None
    gop_size: int | None = Field(default=None, ge=1, le=10_000)
    bframes: int | None = Field(default=None, ge=0, le=7)
    b_ref_mode: Literal["disabled", "each", "middle"] = "middle"
    multipass: Literal["disabled", "qres", "fullres"] = "qres"
    spatial_aq: bool = True
    temporal_aq: bool = True
    aq_strength: int = Field(default=8, ge=1, le=15)
    lookahead: int = Field(default=20, ge=0, le=250)
    extra_args: list[str] = Field(default_factory=list, description="Advanced: raw FFmpeg output options.")

    bitrate_bps: int | None = Field(default=None, exclude=True)
    max_bitrate_bps: int | None = Field(default=None, exclude=True)
    buffer_size_bits: int | None = Field(default=None, exclude=True)

    @field_validator("codec", mode="before")
    @classmethod
    def _codec_alias(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("video.codec must be a string")
        key = value.strip().lower()
        if key not in _CODEC_ALIASES:
            raise ValueError(f"video.codec {value!r} is not supported; use h264, hevc or av1")
        return _CODEC_ALIASES[key]

    @field_validator("preset", mode="before")
    @classmethod
    def _preset_lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("profile", "level", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("extra_args")
    @classmethod
    def _extra_args(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError("video.extra_args must be a list of non-empty strings")
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> VideoSpec:
        if (self.width or self.height) and (self.max_width or self.max_height):
            raise ValueError("video: use either width/height or max_width/max_height, not both")
        for name in ("width", "height", "max_width", "max_height"):
            val = getattr(self, name)
            if val is not None and val % 2:
                raise ValueError(f"video.{name} must be an even number (got {val})")
        bitrate = parse_bitrate(self.bitrate, field="video.bitrate") if self.bitrate is not None else None
        max_bitrate = (
            parse_bitrate(self.max_bitrate, field="video.max_bitrate") if self.max_bitrate is not None else None
        )
        buffer_size = (
            parse_bitrate(self.buffer_size, field="video.buffer_size") if self.buffer_size is not None else None
        )
        if self.rate_control in ("vbr", "cbr") and bitrate is None:
            raise ValueError(f"video.bitrate is required when rate_control is '{self.rate_control}'")
        if self.rate_control == "cq" and bitrate is not None:
            raise ValueError(
                "video.bitrate is ignored in 'cq' mode; set rate_control to 'vbr' or 'cbr' "
                "(or use max_bitrate to cap peaks in cq mode)"
            )
        if bitrate and max_bitrate and max_bitrate < bitrate:
            raise ValueError("video.max_bitrate must be >= video.bitrate")
        if self.codec != "av1" and self.cq > 51:
            raise ValueError("video.cq must be between 0 and 51 for h264/hevc")
        if self.profile is not None and self.profile not in CODEC_PROFILES[self.codec]:
            raise ValueError(
                f"video.profile {self.profile!r} is not valid for {self.codec}; "
                f"choose one of {sorted(CODEC_PROFILES[self.codec])}"
            )
        if self.pixel_format in ("p010le", "yuv444p") and self.codec == "h264" and self.pixel_format == "p010le":
            raise ValueError("h264 does not support 10-bit (p010le); use hevc or av1")
        if self.pixel_format == "yuv444p" and self.codec == "av1":
            raise ValueError("av1_nvenc does not support 4:4:4 (yuv444p)")
        if self.tune == "lossless" and self.rate_control != "cq":
            raise ValueError("video.tune 'lossless' only works with rate_control 'cq'")
        object.__setattr__(self, "bitrate_bps", bitrate)
        object.__setattr__(self, "max_bitrate_bps", max_bitrate)
        object.__setattr__(self, "buffer_size_bits", buffer_size)
        return self

    @property
    def effective_pixel_format(self) -> str:
        if self.pixel_format:
            return self.pixel_format
        if self.profile == "main10":
            return "p010le"
        return "yuv420p"

    @property
    def effective_profile(self) -> str:
        if self.profile:
            return self.profile
        if self.codec == "h264":
            return "high444p" if self.effective_pixel_format == "yuv444p" else "high"
        if self.codec == "hevc":
            if self.effective_pixel_format == "p010le":
                return "main10"
            if self.effective_pixel_format == "yuv444p":
                return "rext"
            return "main"
        return "main"


class AudioSpec(StrictModel):
    codec: Literal["aac", "copy", "none"] = "aac"
    bitrate: str | int = "128k"
    channels: int | None = Field(default=None, ge=1, le=8)
    sample_rate: int | None = Field(default=None, ge=8000, le=192_000)

    bitrate_bps: int = Field(default=128_000, exclude=True)

    @field_validator("codec", mode="before")
    @classmethod
    def _codec(cls, value: Any) -> Any:
        if value is None or value is False:
            return "none"
        if isinstance(value, str):
            v = value.strip().lower()
            return {"off": "none", "no": "none", "mute": "none", "disabled": "none"}.get(v, v)
        return value

    @model_validator(mode="after")
    def _normalise(self) -> AudioSpec:
        object.__setattr__(self, "bitrate_bps", parse_bitrate(self.bitrate, field="audio.bitrate"))
        return self


class OutputSpec(StrictModel):
    """Where the encoded file goes.

    - ``s3``: upload with the worker's S3 credentials (env). ``bucket``/``key`` optional.
    - ``presigned_put``: HTTP PUT to a pre-signed upload URL you generated (no credentials on the worker).
    - ``local``: write to a path inside the container (network volume or local testing).
    - ``base64``: return the bytes inline (small test files only, capped by MAX_INLINE_OUTPUT_BYTES).
    """

    type: Literal["s3", "presigned_put", "local", "base64"] = "s3"
    bucket: str | None = None
    key: str | None = None
    presign_ttl: int | None = Field(default=None, ge=0, le=7 * 24 * 3600)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    path: str | None = None
    overwrite: bool = False
    content_type: str | None = None

    @model_validator(mode="after")
    def _checks(self) -> OutputSpec:
        if self.type == "presigned_put":
            if not self.url or not re.match(r"^https?://", self.url):
                raise ValueError("output.url (http(s) pre-signed PUT URL) is required for type 'presigned_put'")
        if self.type == "local" and not self.path:
            raise ValueError("output.path is required for type 'local'")
        if self.type == "s3" and self.key is not None:
            if self.key.startswith("/") or ".." in self.key.split("/"):
                raise ValueError("output.key must be a relative object key without '..' segments")
        if self.type != "s3" and (self.bucket or self.key or self.presign_ttl is not None):
            raise ValueError("output.bucket/key/presign_ttl are only valid for type 's3'")
        if self.type != "presigned_put" and (self.url or self.headers):
            raise ValueError("output.url/headers are only valid for type 'presigned_put'")
        if self.type != "local" and self.path:
            raise ValueError("output.path is only valid for type 'local'")
        return self


class OptionsSpec(StrictModel):
    hw_decode: Literal["auto", "on", "off"] = Field(
        default="auto", description="GPU (NVDEC) decoding: auto tries GPU then falls back to CPU decode."
    )
    fallback_to_software: bool = Field(
        default=False, description="If NVENC fails, retry with CPU encoders (slow; reported in the result)."
    )
    timeout_seconds: int | None = Field(default=None, ge=1, le=24 * 3600, description="Per-encode FFmpeg timeout.")
    verify_output: bool = True
    faststart: bool = Field(default=True, description="mp4/mov: move moov atom to the front for streaming.")
    progress_updates: bool = True
    keep_metadata: bool = Field(default=True, description="Copy container-level metadata tags from the source.")


class ClipSpec(StrictModel):
    """One output segment when producing several clips (e.g. shorts) from one source."""

    name: str
    trim: TrimSpec
    aspect: AspectSpec | None = None
    output: OutputSpec | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _SAFE_NAME_RE.match(value):
            raise ValueError(
                f"clip name {value!r} must be 1-128 chars of letters, digits, '.', '_' or '-' (no spaces)"
            )
        return value


class JobInput(StrictModel):
    input_url: str = Field(
        validation_alias=AliasChoices("input_url", "url", "source_url", "video_url", "input_path"),
        description="https:// (signed) URL, s3://bucket/key, or an absolute path inside the container.",
    )
    input_headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers for the download.")
    name: str | None = Field(default=None, description="Base name for the output file (no extension).")
    container: Container = "mp4"
    video: VideoSpec = Field(default_factory=VideoSpec)
    audio: AudioSpec = Field(default_factory=AudioSpec)
    trim: TrimSpec | None = None
    aspect: AspectSpec | None = None
    clips: list[ClipSpec] | None = None
    output: OutputSpec | None = None
    options: OptionsSpec = Field(default_factory=OptionsSpec)

    @field_validator("container", mode="before")
    @classmethod
    def _container(cls, value: Any) -> Any:
        return value.strip().lower().lstrip(".") if isinstance(value, str) else value

    @field_validator("input_url")
    @classmethod
    def _input_url(cls, value: str) -> str:
        if not value:
            raise ValueError("input_url must not be empty")
        if value.startswith(("http://", "https://", "s3://", "/")):
            return value
        raise ValueError(
            "input_url must start with https://, http://, s3:// or be an absolute path inside the container"
        )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_NAME_RE.match(value):
            raise ValueError("name must be 1-128 chars of letters, digits, '.', '_' or '-' (no spaces)")
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> JobInput:
        if self.clips is not None:
            if not self.clips:
                raise ValueError("clips must contain at least one clip when provided")
            if self.trim is not None:
                raise ValueError("use either top-level 'trim' or 'clips', not both")
            names = [c.name for c in self.clips]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(f"clip names must be unique (duplicates: {dupes})")
            if len(self.clips) > 100:
                raise ValueError("at most 100 clips per job")
        if self.container == "mov" and self.video.codec == "av1":
            raise ValueError("av1 cannot be stored in a .mov container; use mp4 or mkv")
        return self


def parse_job_input(raw: Any) -> JobInput:
    """Validate the raw ``job["input"]`` dict into a :class:`JobInput`.

    Raises :class:`InputValidationError` with every problem listed.
    """
    if raw is None:
        raise InputValidationError(
            "Job payload has no 'input' object.",
            hint='Send {"input": {"input_url": "https://...", ...}}.',
        )
    if not isinstance(raw, dict):
        raise InputValidationError(f"'input' must be a JSON object, got {type(raw).__name__}.")
    try:
        return JobInput.model_validate(raw)
    except ValidationError as exc:
        problems = []
        for err in exc.errors(include_url=False):
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__") or "input"
            problems.append({"field": loc, "error": err.get("msg", "invalid value")})
        summary = "; ".join(f"{p['field']}: {p['error']}" for p in problems[:5])
        if len(problems) > 5:
            summary += f" (+{len(problems) - 5} more)"
        raise InputValidationError(
            f"Invalid job input: {summary}",
            details={"problems": problems},
            hint="See README 'Job input reference' for the accepted fields.",
        ) from None
