"""Structured, explicit errors.

Every failure path in the service ends up as a :class:`VideoServiceError`
(or a subclass) which is serialised into the job response. Nothing is ever
swallowed silently: the ``code`` is stable and machine-readable, ``message``
is human-readable, ``hint`` says what to do about it, and ``details`` carries
diagnostic data (FFmpeg stderr tail, HTTP status, sizes, ...).
"""

from __future__ import annotations

from typing import Any


class VideoServiceError(Exception):
    """Base class for all errors returned to the caller."""

    code: str = "INTERNAL_ERROR"
    #: Set when the job may succeed if simply retried (transient network, ...).
    retryable: bool = False
    #: Set when the worker itself is suspect (GPU/driver problem). RunPod will
    #: recycle the worker after the job so the next job lands on a healthy one.
    refresh_worker: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
        refresh_worker: bool | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.hint = hint
        self.details: dict[str, Any] = dict(details or {})
        if retryable is not None:
            self.retryable = retryable
        if refresh_worker is not None:
            self.refresh_worker = refresh_worker
        self.stage = stage

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.stage:
            payload["stage"] = self.stage
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"


class InputValidationError(VideoServiceError):
    code = "INVALID_INPUT"


class StorageConfigError(VideoServiceError):
    code = "STORAGE_NOT_CONFIGURED"


class DownloadError(VideoServiceError):
    code = "DOWNLOAD_FAILED"
    retryable = True


class DiskSpaceError(VideoServiceError):
    code = "INSUFFICIENT_DISK_SPACE"


class ProbeError(VideoServiceError):
    code = "PROBE_FAILED"


class UnsupportedMediaError(VideoServiceError):
    code = "UNSUPPORTED_MEDIA"


class EncodeError(VideoServiceError):
    code = "ENCODE_FAILED"


class EncodeTimeoutError(VideoServiceError):
    code = "ENCODE_TIMEOUT"


class GpuError(VideoServiceError):
    """GPU / NVENC / driver problem: recycle the worker after the job."""

    code = "GPU_ERROR"
    refresh_worker = True


class OutputVerificationError(VideoServiceError):
    code = "OUTPUT_VERIFICATION_FAILED"


class UploadError(VideoServiceError):
    code = "UPLOAD_FAILED"
    retryable = True


class OutputExistsError(VideoServiceError):
    code = "OUTPUT_ALREADY_EXISTS"
