"""Input download and output delivery.

Inputs:  https:// (signed) URLs, ``s3://bucket/key`` or a local path.
Outputs: S3-compatible storage (AWS S3, Cloudflare R2, Backblaze B2, MinIO,
         RunPod network-volume S3 API), a pre-signed PUT URL, a local path,
         or (small test files only) inline base64.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .errors import (
    DiskSpaceError,
    DownloadError,
    InputValidationError,
    OutputExistsError,
    StorageConfigError,
    UploadError,
)
from .schema import CONTAINER_CONTENT_TYPE, OutputSpec
from .units import human_bytes, redact_url

log = logging.getLogger(__name__)

CHUNK = 4 * 1024 * 1024
ProgressFn = Callable[[int, int | None], None]


# --------------------------------------------------------------------------
# Source parsing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    kind: str  # http | s3 | local
    url: str
    bucket: str | None = None
    key: str | None = None
    path: str | None = None

    @property
    def display(self) -> str:
        return redact_url(self.url) if self.kind == "http" else self.url

    @property
    def filename(self) -> str:
        if self.kind == "http":
            name = os.path.basename(urlsplit(self.url).path)
        elif self.kind == "s3":
            name = os.path.basename(self.key or "")
        else:
            name = os.path.basename(self.path or "")
        return name or "input"


def parse_source(input_url: str) -> Source:
    if input_url.startswith(("http://", "https://")):
        return Source(kind="http", url=input_url)
    if input_url.startswith("s3://"):
        rest = input_url[len("s3://"):]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise InputValidationError(f"s3 input must look like s3://bucket/key (got {redact_url(input_url)!r}).")
        return Source(kind="s3", url=input_url, bucket=bucket, key=key)
    if input_url.startswith("/"):
        return Source(kind="local", url=input_url, path=input_url)
    raise InputValidationError("input_url must be an http(s) URL, s3:// URI or absolute path.")


# --------------------------------------------------------------------------
# Disk space
# --------------------------------------------------------------------------
def ensure_disk_space(work_dir: str, needed_bytes: int, *, purpose: str) -> None:
    usage = shutil.disk_usage(work_dir)
    if usage.free < needed_bytes:
        raise DiskSpaceError(
            f"Not enough free disk for {purpose}: need about {human_bytes(needed_bytes)}, "
            f"only {human_bytes(usage.free)} free in {work_dir}.",
            details={"needed_bytes": needed_bytes, "free_bytes": usage.free, "work_dir": work_dir},
            hint="Increase the container disk size of the endpoint (2-3x the input size), "
                 "or point WORK_DIR at a network volume.",
        )


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _http_session(retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=retries, connect=retries, read=retries, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET", "HEAD"}),
                  raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_http(url: str, dest: str, settings: Settings, *, headers: dict[str, str] | None = None,
                  progress: ProgressFn | None = None) -> int:
    """Stream ``url`` to ``dest``. Returns the byte count. Retries whole transfers."""
    shown = redact_url(url)
    last_error: Exception | None = None
    session = _http_session(settings.download_attempts)
    for attempt in range(1, settings.download_attempts + 1):
        try:
            with session.get(url, headers=headers or {}, stream=True,
                             timeout=(15, settings.download_timeout_seconds), allow_redirects=True) as resp:
                if resp.status_code >= 400:
                    body = resp.content[:600].decode("utf-8", "replace")
                    raise DownloadError(
                        f"Download of {shown} failed with HTTP {resp.status_code}.",
                        details={"status_code": resp.status_code, "body_snippet": body},
                        retryable=resp.status_code in (408, 429, 500, 502, 503, 504),
                        hint="For 403/404 the signed URL is probably expired, wrong, or unsigned.",
                    )
                content_type = (resp.headers.get("Content-Type") or "").lower()
                length_header = resp.headers.get("Content-Length")
                total = int(length_header) if length_header and length_header.isdigit() else None
                if total is not None:
                    if total > settings.max_input_bytes:
                        raise DownloadError(
                            f"Input is {human_bytes(total)}, above MAX_INPUT_BYTES ({human_bytes(settings.max_input_bytes)}).",
                            code="INPUT_TOO_LARGE", retryable=False,
                        )
                    ensure_disk_space(os.path.dirname(dest), int(total * settings.disk_space_safety_factor),
                                      purpose="download + encode")
                if content_type.startswith(("text/html", "text/plain", "application/json", "application/xml")) \
                        and (total is None or total < 1_000_000):
                    snippet = next(resp.iter_content(2000), b"")[:600].decode("utf-8", "replace")
                    raise DownloadError(
                        f"Download of {shown} returned {content_type!r} instead of a video.",
                        details={"content_type": content_type, "body_snippet": snippet}, retryable=False,
                        hint="The URL points to an error/HTML page; check the signed URL.",
                    )
                written = 0
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        if written > settings.max_input_bytes:
                            raise DownloadError(
                                f"Input exceeded MAX_INPUT_BYTES ({human_bytes(settings.max_input_bytes)}) during download.",
                                code="INPUT_TOO_LARGE", retryable=False,
                            )
                        if progress:
                            progress(written, total)
                if total is not None and written != total:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"incomplete read: got {written} of {total} bytes")
                if written == 0:
                    raise DownloadError(f"Download of {shown} returned an empty body.", retryable=False)
                return written
        except DownloadError as exc:
            if not exc.retryable or attempt == settings.download_attempts:
                raise
            last_error = exc
        except (requests.exceptions.RequestException, OSError) as exc:
            last_error = exc
            if isinstance(exc, OSError) and not isinstance(exc, requests.exceptions.RequestException) \
                    and getattr(exc, "errno", None) == 28:
                raise DiskSpaceError(f"Disk full while downloading {shown}.", details={"path": dest}) from exc
        log.warning("Download attempt %d/%d for %s failed: %s", attempt, settings.download_attempts, shown, last_error)
        time.sleep(min(2 ** attempt, 15))
    raise DownloadError(
        f"Download of {shown} failed after {settings.download_attempts} attempts: {last_error}",
        details={"last_error": str(last_error)},
        hint="Check the URL is reachable from RunPod and the signed URL has not expired.",
    )


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------
def s3_client(settings: Settings):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise StorageConfigError("boto3 is not installed.", hint="pip install boto3") from exc

    s3 = settings.s3
    config = Config(
        signature_version="s3v4",
        retries={"max_attempts": 5, "mode": "standard"},
        s3={"addressing_style": "path" if s3.force_path_style else "auto"},
    )
    kwargs: dict[str, Any] = {"config": config}
    if s3.endpoint_url:
        kwargs["endpoint_url"] = s3.endpoint_url
    if s3.region:
        kwargs["region_name"] = s3.region
    if s3.has_explicit_credentials:
        kwargs["aws_access_key_id"] = s3.access_key_id
        kwargs["aws_secret_access_key"] = s3.secret_access_key
        if s3.session_token:
            kwargs["aws_session_token"] = s3.session_token
    return boto3.client("s3", **kwargs)


def _s3_error(exc: Exception, action: str, bucket: str, key: str) -> UploadError | DownloadError | StorageConfigError:
    name = type(exc).__name__
    text = str(exc)
    if name in ("NoCredentialsError", "PartialCredentialsError"):
        return StorageConfigError(
            f"Cannot {action} s3://{bucket}/{key}: no S3 credentials configured.",
            hint="Set S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY (and S3_ENDPOINT_URL for non-AWS storage) "
                 "in the RunPod endpoint environment variables.",
        )
    code = None
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
    hint = None
    if code in ("AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "403"):
        hint = "The S3 credentials are wrong or lack permission on this bucket."
    elif code in ("NoSuchBucket", "404", "NoSuchKey"):
        hint = "Bucket or key does not exist (check S3_BUCKET / the key / S3_ENDPOINT_URL and region)."
    elif "endpoint" in text.lower() or name == "EndpointConnectionError":
        hint = "Cannot reach the S3 endpoint; check S3_ENDPOINT_URL and S3_REGION."
    cls = DownloadError if action == "download" else UploadError
    return cls(f"Failed to {action} s3://{bucket}/{key}: {name}: {text[:400]}",
               details={"s3_error_code": code, "exception": name}, hint=hint,
               retryable=code not in ("AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "NoSuchBucket", "NoSuchKey"))


def download_s3(bucket: str, key: str, dest: str, settings: Settings, *, progress: ProgressFn | None = None) -> int:
    client = s3_client(settings)
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        total = int(head.get("ContentLength") or 0)
        if total > settings.max_input_bytes:
            raise DownloadError(
                f"Input is {human_bytes(total)}, above MAX_INPUT_BYTES ({human_bytes(settings.max_input_bytes)}).",
                code="INPUT_TOO_LARGE", retryable=False,
            )
        if total:
            ensure_disk_space(os.path.dirname(dest), int(total * settings.disk_space_safety_factor),
                              purpose="download + encode")
        done = 0

        def _cb(n: int) -> None:
            nonlocal done
            done += n
            if progress:
                progress(done, total or None)

        client.download_file(bucket, key, dest, Callback=_cb)
    except (DownloadError, DiskSpaceError):
        raise
    except Exception as exc:  # botocore raises many classes; normalise them
        raise _s3_error(exc, "download", bucket, key) from exc
    return os.path.getsize(dest)


def s3_object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        response = getattr(exc, "response", None)
        code = (response.get("Error") or {}).get("Code") if isinstance(response, dict) else None
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode") if isinstance(response, dict) else None
        if code in ("404", "NoSuchKey", "NotFound") or status == 404:
            return False
        raise _s3_error(exc, "upload", bucket, key) from exc


def upload_s3(path: str, bucket: str, key: str, settings: Settings, *, content_type: str,
              overwrite: bool, presign_ttl: int | None) -> dict[str, Any]:
    from boto3.s3.transfer import TransferConfig

    client = s3_client(settings)
    if not overwrite and s3_object_exists(client, bucket, key):
        raise OutputExistsError(
            f"s3://{bucket}/{key} already exists and output.overwrite is false.",
            hint="Set output.overwrite=true or choose a different output.key.",
        )
    size = os.path.getsize(path)
    try:
        client.upload_file(
            path, bucket, key, ExtraArgs={"ContentType": content_type},
            Config=TransferConfig(multipart_threshold=64 * 1024 * 1024, multipart_chunksize=64 * 1024 * 1024,
                                  max_concurrency=4),
        )
    except Exception as exc:
        raise _s3_error(exc, "upload", bucket, key) from exc

    result: dict[str, Any] = {"type": "s3", "bucket": bucket, "key": key, "uri": f"s3://{bucket}/{key}",
                              "size_bytes": size, "content_type": content_type}
    ttl = settings.s3.presign_ttl_seconds if presign_ttl is None else presign_ttl
    if ttl and ttl > 0:
        try:
            result["url"] = client.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=int(ttl))
            result["url_expires_in_seconds"] = int(ttl)
        except Exception as exc:  # pragma: no cover - presign is offline, but stay explicit
            result["presign_error"] = f"{type(exc).__name__}: {exc}"
    if settings.s3.public_base_url:
        result["public_url"] = settings.s3.public_base_url.rstrip("/") + "/" + key
    return result


# --------------------------------------------------------------------------
# Other destinations
# --------------------------------------------------------------------------
def upload_presigned_put(path: str, url: str, *, content_type: str, headers: dict[str, str]) -> dict[str, Any]:
    size = os.path.getsize(path)
    send_headers = {"Content-Type": content_type, "Content-Length": str(size)}
    send_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with open(path, "rb") as fh:
                resp = requests.put(url, data=fh, headers=send_headers, timeout=(15, 600))
            if 200 <= resp.status_code < 300:
                return {"type": "presigned_put", "url": redact_url(url), "size_bytes": size,
                        "status_code": resp.status_code, "etag": resp.headers.get("ETag")}
            body = resp.text[:600]
            retryable = resp.status_code in (408, 429, 500, 502, 503, 504)
            last_error = UploadError(
                f"PUT to {redact_url(url)} failed with HTTP {resp.status_code}.",
                details={"status_code": resp.status_code, "body_snippet": body}, retryable=retryable,
                hint="Check the pre-signed URL has not expired and was signed for PUT with this Content-Type.",
            )
            if not retryable:
                raise last_error
        except requests.exceptions.RequestException as exc:
            last_error = exc
        log.warning("Upload attempt %d failed: %s", attempt, last_error)
        time.sleep(2 ** attempt)
    if isinstance(last_error, UploadError):
        raise last_error
    raise UploadError(f"PUT to {redact_url(url)} failed: {last_error}", details={"last_error": str(last_error)})


def copy_local(path: str, dest: str, *, overwrite: bool) -> dict[str, Any]:
    dest = os.path.abspath(dest)
    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(path))
    if os.path.exists(dest) and not overwrite:
        raise OutputExistsError(f"{dest} already exists and output.overwrite is false.",
                                hint="Set output.overwrite=true or choose another output.path.")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        shutil.copyfile(path, dest)
    except OSError as exc:
        raise UploadError(f"Could not write output to {dest}: {exc}", retryable=False) from exc
    return {"type": "local", "path": dest, "size_bytes": os.path.getsize(dest)}


def inline_base64(path: str, settings: Settings) -> dict[str, Any]:
    size = os.path.getsize(path)
    if size > settings.max_inline_output_bytes:
        raise UploadError(
            f"Output is {human_bytes(size)}; inline base64 output is capped at "
            f"{human_bytes(settings.max_inline_output_bytes)} (MAX_INLINE_OUTPUT_BYTES).",
            code="OUTPUT_TOO_LARGE_FOR_INLINE", retryable=False,
            hint="Use output.type 's3' or 'presigned_put' for real files.",
        )
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return {"type": "base64", "size_bytes": size, "data_base64": data}


# --------------------------------------------------------------------------
# Output resolution and delivery
# --------------------------------------------------------------------------
def resolve_output(spec: OutputSpec | None, settings: Settings, *, job_id: str, filename: str,
                   container: str) -> OutputSpec:
    """Fill in defaults (bucket/key) and validate the destination is usable *before* encoding."""
    if spec is None:
        if not settings.s3.is_configured:
            raise StorageConfigError(
                "No output destination given and S3_BUCKET is not configured on the worker.",
                hint="Add an 'output' object to the job (type s3 / presigned_put / local / base64) or set "
                     "S3_BUCKET (+ credentials) in the endpoint environment variables.",
            )
        spec = OutputSpec(type="s3")
    if spec.type == "s3":
        bucket = spec.bucket or settings.s3.bucket
        if not bucket:
            raise StorageConfigError(
                "output.type is 's3' but no bucket was given and S3_BUCKET is not set.",
                hint="Set output.bucket in the job or S3_BUCKET in the endpoint environment.",
            )
        key = spec.key or f"{settings.s3.output_prefix}{job_id}/{filename}"
        return spec.model_copy(update={"bucket": bucket, "key": key})
    return spec


def deliver(path: str, spec: OutputSpec, settings: Settings, *, container: str) -> dict[str, Any]:
    content_type = spec.content_type or CONTAINER_CONTENT_TYPE.get(container, "application/octet-stream")
    if spec.type == "s3":
        assert spec.bucket and spec.key
        return upload_s3(path, spec.bucket, spec.key, settings, content_type=content_type,
                         overwrite=spec.overwrite, presign_ttl=spec.presign_ttl)
    if spec.type == "presigned_put":
        assert spec.url
        return upload_presigned_put(path, spec.url, content_type=content_type, headers=spec.headers)
    if spec.type == "local":
        assert spec.path
        return copy_local(path, spec.path, overwrite=spec.overwrite)
    return inline_base64(path, settings)


def unique_suffix() -> str:
    return uuid.uuid4().hex[:8]
