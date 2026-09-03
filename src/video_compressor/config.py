"""Runtime settings, all sourced from environment variables.

Nothing sensitive is ever hard-coded. See README for the full list.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from .errors import InputValidationError

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


def env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise InputValidationError(
        f"Environment variable {key}={raw!r} is not a boolean (use true/false).",
        code="INVALID_CONFIGURATION",
    )


def env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise InputValidationError(
            f"Environment variable {key}={raw!r} is not an integer.",
            code="INVALID_CONFIGURATION",
        ) from exc


def _env_str(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    raw = env.get(key)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw != "" else default


@dataclass(frozen=True)
class S3Settings:
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    region: str | None = None
    bucket: str | None = None
    output_prefix: str = "compressed/"
    presign_ttl_seconds: int = 3600
    force_path_style: bool = False
    public_base_url: str | None = None

    @property
    def has_explicit_credentials(self) -> bool:
        return bool(self.access_key_id and self.secret_access_key)

    @property
    def is_configured(self) -> bool:
        """True when uploads can be attempted without per-job bucket info."""
        return bool(self.bucket)


@dataclass(frozen=True)
class Settings:
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    work_dir: str = "/tmp/video-jobs"
    keep_work_dir: bool = False
    #: "nvenc" (strict: fail if NVENC unavailable), "software" (CPU encoders
    #: only, for laptops/CI) or "auto" (NVENC if available, otherwise CPU).
    encoder_backend: str = "nvenc"
    ffmpeg_timeout_seconds: int = 0
    max_input_bytes: int = 20_000_000_000
    download_timeout_seconds: int = 60
    download_attempts: int = 3
    max_inline_output_bytes: int = 8_000_000
    disk_space_safety_factor: float = 2.2
    progress_interval_seconds: float = 5.0
    log_level: str = "INFO"
    s3: S3Settings = field(default_factory=S3Settings)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        backend = (_env_str(env, "ENCODER_BACKEND", "nvenc") or "nvenc").lower()
        if backend not in {"nvenc", "software", "auto"}:
            raise InputValidationError(
                f"ENCODER_BACKEND={backend!r} is invalid; expected nvenc, software or auto.",
                code="INVALID_CONFIGURATION",
            )
        s3 = S3Settings(
            endpoint_url=_env_str(env, "S3_ENDPOINT_URL") or _env_str(env, "AWS_ENDPOINT_URL"),
            access_key_id=_env_str(env, "S3_ACCESS_KEY_ID") or _env_str(env, "AWS_ACCESS_KEY_ID"),
            secret_access_key=_env_str(env, "S3_SECRET_ACCESS_KEY")
            or _env_str(env, "AWS_SECRET_ACCESS_KEY"),
            session_token=_env_str(env, "S3_SESSION_TOKEN") or _env_str(env, "AWS_SESSION_TOKEN"),
            region=_env_str(env, "S3_REGION") or _env_str(env, "AWS_REGION") or _env_str(env, "AWS_DEFAULT_REGION"),
            bucket=_env_str(env, "S3_BUCKET"),
            output_prefix=_env_str(env, "S3_OUTPUT_PREFIX", "compressed/") or "",
            presign_ttl_seconds=env_int(env, "S3_PRESIGN_TTL", 3600),
            force_path_style=env_bool(env, "S3_FORCE_PATH_STYLE", False),
            public_base_url=_env_str(env, "S3_PUBLIC_BASE_URL"),
        )
        return cls(
            ffmpeg_bin=_env_str(env, "FFMPEG_BIN", "ffmpeg") or "ffmpeg",
            ffprobe_bin=_env_str(env, "FFPROBE_BIN", "ffprobe") or "ffprobe",
            work_dir=_env_str(env, "WORK_DIR", "/tmp/video-jobs") or "/tmp/video-jobs",
            keep_work_dir=env_bool(env, "KEEP_WORK_DIR", False),
            encoder_backend=backend,
            ffmpeg_timeout_seconds=env_int(env, "FFMPEG_TIMEOUT_SECONDS", 0),
            max_input_bytes=env_int(env, "MAX_INPUT_BYTES", 20_000_000_000),
            download_timeout_seconds=env_int(env, "DOWNLOAD_TIMEOUT_SECONDS", 60),
            download_attempts=max(1, env_int(env, "DOWNLOAD_ATTEMPTS", 3)),
            max_inline_output_bytes=env_int(env, "MAX_INLINE_OUTPUT_BYTES", 8_000_000),
            log_level=(_env_str(env, "LOG_LEVEL", "INFO") or "INFO").upper(),
            s3=s3,
        )

    def redacted(self) -> dict:
        """Settings safe for logging (no secrets)."""
        return {
            "ffmpeg_bin": self.ffmpeg_bin,
            "work_dir": self.work_dir,
            "encoder_backend": self.encoder_backend,
            "ffmpeg_timeout_seconds": self.ffmpeg_timeout_seconds,
            "max_input_bytes": self.max_input_bytes,
            "s3_endpoint_url": self.s3.endpoint_url,
            "s3_bucket": self.s3.bucket,
            "s3_output_prefix": self.s3.output_prefix,
            "s3_region": self.s3.region,
            "s3_has_credentials": self.s3.has_explicit_credentials,
            "s3_force_path_style": self.s3.force_path_style,
        }
