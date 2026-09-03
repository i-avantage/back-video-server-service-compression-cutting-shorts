"""Job orchestration: validate -> fetch -> probe -> encode -> verify -> deliver.

``process_job`` never raises for expected failures: it always returns a
dict, with ``status`` = ``"success"`` or ``"error"``. When ``"error"`` is
present, RunPod marks the job FAILED and exposes the structured details.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import __version__
from .capabilities import Capabilities, detect_capabilities
from .config import Settings
from .encoder import EncodeOutcome, encode_with_fallback
from .errors import InputValidationError, OutputVerificationError, VideoServiceError
from .ffmpeg_cmd import RenderSpec
from .probe import MediaInfo, probe_media
from .schema import CONTAINER_EXTENSION, JobInput, TrimSpec, parse_job_input
from .storage import (
    Source,
    deliver,
    download_http,
    download_s3,
    parse_source,
    resolve_output,
    unique_suffix,
)
from .units import human_bytes

log = logging.getLogger(__name__)

ProgressHook = Callable[[dict[str, Any]], None]
_LOGGING_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    else:
        root.setLevel(level)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True


def running_on_runpod() -> bool:
    return os.environ.get("RUNPOD_WEBHOOK_GET_JOB") is not None


def _runpod_progress(job: dict[str, Any]) -> ProgressHook | None:
    if not running_on_runpod():
        return None
    try:
        import runpod  # type: ignore
    except ImportError:  # pragma: no cover
        return None

    def hook(payload: dict[str, Any]) -> None:
        runpod.serverless.progress_update(job, payload)

    return hook


@dataclass
class Timings:
    started: float = field(default_factory=time.monotonic)
    values: dict[str, float] = field(default_factory=dict)

    def add(self, key: str, seconds: float) -> None:
        self.values[key] = round(self.values.get(key, 0.0) + seconds, 3)

    def to_dict(self) -> dict[str, float]:
        out = dict(self.values)
        out["total_seconds"] = round(time.monotonic() - self.started, 3)
        return out


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:100] or "video"


def _worker_info(settings: Settings, caps: Capabilities | None) -> dict[str, Any]:
    info: dict[str, Any] = {"service_version": __version__, "encoder_backend": settings.encoder_backend}
    if caps:
        info.update({"gpu": caps.gpu_name, "driver_version": caps.driver_version,
                     "ffmpeg_version": caps.ffmpeg_version, "nvenc_available": caps.nvenc_available})
    return info


def build_render_specs(spec: JobInput, settings: Settings, *, job_id: str, source_name: str) -> list[RenderSpec]:
    """Expand the job into one RenderSpec per output (single file or N clips)."""
    base = spec.name or f"{_safe_stem(source_name)}_compressed"
    ext = CONTAINER_EXTENSION[spec.container]
    renders: list[RenderSpec] = []
    if spec.clips:
        for clip in spec.clips:
            filename = f"{base}_{clip.name}.{ext}"
            output = clip.output or _derive_clip_output(spec, clip.name, ext)
            renders.append(RenderSpec(
                name=clip.name, filename=filename, container=spec.container, video=spec.video,
                audio=spec.audio, options=spec.options, trim=clip.trim,
                aspect=clip.aspect if clip.aspect is not None else spec.aspect,
                output=resolve_output(output, settings, job_id=job_id, filename=filename, container=spec.container),
            ))
    else:
        filename = f"{base}.{ext}"
        renders.append(RenderSpec(
            name=base, filename=filename, container=spec.container, video=spec.video, audio=spec.audio,
            options=spec.options, trim=spec.trim, aspect=spec.aspect,
            output=resolve_output(spec.output, settings, job_id=job_id, filename=filename, container=spec.container),
        ))
    return renders


def _derive_clip_output(spec: JobInput, clip_name: str, ext: str):
    """Clips without their own output inherit the job output with a per-clip key/path."""
    out = spec.output
    if out is None:
        return None
    if out.type == "s3" and out.key:
        stem, key_ext = os.path.splitext(out.key)
        return out.model_copy(update={"key": f"{stem}_{clip_name}{key_ext or '.' + ext}"})
    if out.type == "local" and out.path and not os.path.isdir(out.path):
        stem, path_ext = os.path.splitext(out.path)
        return out.model_copy(update={"path": f"{stem}_{clip_name}{path_ext or '.' + ext}"})
    if out.type == "presigned_put":
        raise InputValidationError(
            "A single presigned_put URL cannot receive several clips; give each clip its own 'output'.",
        )
    return out


def _check_trim(trim: TrimSpec | None, media: MediaInfo, name: str, warnings: list[str]) -> TrimSpec | None:
    if trim is None:
        return None
    if trim.start_seconds >= media.duration:
        raise InputValidationError(
            f"{name}: trim.start ({trim.start_seconds}s) is beyond the media duration ({media.duration:.3f}s).",
        )
    if trim.end_seconds is not None and trim.end_seconds > media.duration + 0.05:
        warnings.append(
            f"{name}: trim end {trim.end_seconds}s exceeds media duration {media.duration:.3f}s; clamped to the end."
        )
        return TrimSpec.model_validate({"start": trim.start_seconds, "end": media.duration})
    return trim


def fetch_source(source: Source, dest_dir: str, spec: JobInput, settings: Settings,
                 progress: ProgressHook | None) -> str:
    if source.kind == "local":
        path = source.path or ""
        if not os.path.isfile(path):
            raise InputValidationError(f"Local input {path!r} does not exist inside the container.",
                                       hint="Mount a network volume or use an https:// / s3:// input.")
        if not os.access(path, os.R_OK):
            raise InputValidationError(f"Local input {path!r} is not readable.")
        return path
    ext = os.path.splitext(source.filename)[1][:8] or ".bin"
    dest = os.path.join(dest_dir, f"input{ext}")
    last_emit = [0.0]

    def _dl_progress(done: int, total: int | None) -> None:
        if progress and time.monotonic() - last_emit[0] >= settings.progress_interval_seconds:
            last_emit[0] = time.monotonic()
            payload: dict[str, Any] = {"stage": "downloading", "bytes": done}
            if total:
                payload["total_bytes"] = total
                payload["percent"] = round(100.0 * done / total, 1)
            progress(payload)

    if source.kind == "http":
        download_http(source.url, dest, settings, headers=spec.input_headers, progress=_dl_progress)
    else:
        assert source.bucket and source.key
        download_s3(source.bucket, source.key, dest, settings, progress=_dl_progress)
    return dest


def verify_output(settings: Settings, path: str, render: RenderSpec, media: MediaInfo,
                  outcome: EncodeOutcome) -> MediaInfo:
    """ffprobe the result and check it is a complete, sane file. Never trust exit code 0 alone."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise OutputVerificationError(f"FFmpeg produced no output file for {render.name!r}.",
                                      details={"path": path, "attempts": [a.to_dict() for a in outcome.attempts]})
    try:
        info = probe_media(settings.ffprobe_bin, path)
    except VideoServiceError as exc:
        raise OutputVerificationError(
            f"The encoded file for {render.name!r} is not readable by ffprobe: {exc.message}",
            details=exc.details,
        ) from exc
    expected = render.expected_duration
    if expected is None:
        expected = media.duration - (render.trim.start_seconds if render.trim else 0.0)
    tolerance = max(0.5, 0.02 * expected)
    if abs(info.duration - expected) > tolerance:
        raise OutputVerificationError(
            f"Output duration for {render.name!r} is {info.duration:.3f}s but {expected:.3f}s was expected "
            f"(tolerance {tolerance:.2f}s). The encode may have stopped early.",
            details={"expected_seconds": round(expected, 3), "actual_seconds": round(info.duration, 3),
                     "attempts": [a.to_dict() for a in outcome.attempts], "last_progress": outcome.last_progress},
            hint="Retry the job; if it persists, run with options.hw_decode='off' and report the FFmpeg log.",
        )
    if info.video and (info.video.width, info.video.height) != outcome.geometry_out:
        raise OutputVerificationError(
            f"Output resolution for {render.name!r} is {info.video.width}x{info.video.height}, "
            f"expected {outcome.geometry_out[0]}x{outcome.geometry_out[1]}.",
        )
    if render.audio.codec != "none" and media.audio is not None and info.audio is None:
        raise OutputVerificationError(f"Output for {render.name!r} lost its audio stream.")
    return info


def _output_record(render: RenderSpec, media: MediaInfo, out_info: MediaInfo, outcome: EncodeOutcome,
                   destination: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    src_equiv = media.size_bytes * (out_info.duration / media.duration) if media.duration else media.size_bytes
    ratio = (src_equiv / out_info.size_bytes) if out_info.size_bytes else None
    record: dict[str, Any] = {
        "name": render.name,
        "filename": render.filename,
        "container": render.container,
        "video_codec": render.video.codec,
        "encoder": outcome.encoder,
        "pipeline": outcome.pipeline.value,
        "width": out_info.video.width if out_info.video else None,
        "height": out_info.video.height if out_info.video else None,
        "fps": round(out_info.video.fps, 3) if out_info.video and out_info.video.fps else None,
        "duration_seconds": round(out_info.duration, 3),
        "size_bytes": out_info.size_bytes,
        "bit_rate": out_info.bit_rate,
        "audio_codec": out_info.audio.codec if out_info.audio else None,
        "source_equivalent_bytes": int(src_equiv),
        "compression_ratio": round(ratio, 2) if ratio else None,
        "size_reduction_percent": round(100.0 * (1 - out_info.size_bytes / src_equiv), 1) if src_equiv else None,
        "encode_seconds": round(outcome.elapsed_seconds, 2),
        "encode_speed_x": round(out_info.duration / outcome.elapsed_seconds, 2) if outcome.elapsed_seconds else None,
        "attempts": [a.to_dict() for a in outcome.attempts],
        "destination": destination,
    }
    if warnings:
        record["warnings"] = warnings
    return record


def process_job(job: dict[str, Any], *, progress_hook: ProgressHook | None = None) -> dict[str, Any]:
    """Entry point used by ``handler.py``. Always returns a dict; never raises for job errors."""
    settings: Settings | None = None
    caps: Capabilities | None = None
    timings = Timings()
    job_id = str((job or {}).get("id") or f"local-{unique_suffix()}")
    workspace: str | None = None
    try:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        log.info("[%s] job received", job_id)
        spec = parse_job_input((job or {}).get("input"))
        progress = progress_hook or _runpod_progress(job)
        caps = detect_capabilities(settings)
        source = parse_source(spec.input_url)
        renders = build_render_specs(spec, settings, job_id=job_id, source_name=source.filename)
        log.info("[%s] source=%s outputs=%s backend=%s", job_id, source.display,
                 [r.filename for r in renders], settings.encoder_backend)

        workspace = os.path.join(settings.work_dir, f"{_safe_stem(job_id)}-{unique_suffix()}")
        os.makedirs(workspace, exist_ok=True)

        t = time.monotonic()
        input_path = fetch_source(source, workspace, spec, settings, progress)
        timings.add("download_seconds", time.monotonic() - t)

        t = time.monotonic()
        media = probe_media(settings.ffprobe_bin, input_path)
        timings.add("probe_seconds", time.monotonic() - t)
        log.info("[%s] input: %s", job_id, media.summary())

        job_warnings: list[str] = []
        outputs: list[dict[str, Any]] = []
        for index, render in enumerate(renders):
            render_warnings: list[str] = []
            render.trim = _check_trim(render.trim, media, render.name, render_warnings)
            out_path = os.path.join(workspace, render.filename)

            def _enc_progress(p: dict[str, Any], _render: RenderSpec = render, _i: int = index) -> None:
                _name = _render.name
                if progress and _render.options.progress_updates:
                    payload = {"stage": "encoding", "output": _name, "output_index": _i + 1,
                               "outputs_total": len(renders)}
                    payload.update({k: p[k] for k in ("percent", "fps", "speed", "out_time_seconds", "frame") if k in p})
                    progress(payload)

            t = time.monotonic()
            outcome = encode_with_fallback(settings, caps, render, media, input_path, out_path,
                                           progress_cb=_enc_progress, log_dir=workspace)
            timings.add("encode_seconds", time.monotonic() - t)
            render_warnings.extend(outcome.warnings)

            t = time.monotonic()
            out_info = verify_output(settings, out_path, render, media, outcome) if render.options.verify_output \
                else probe_media(settings.ffprobe_bin, out_path)
            timings.add("verify_seconds", time.monotonic() - t)

            if progress:
                progress({"stage": "uploading", "output": render.name, "output_index": index + 1,
                          "outputs_total": len(renders), "size_bytes": out_info.size_bytes})
            t = time.monotonic()
            destination = deliver(out_path, render.output, settings, container=render.container)
            timings.add("upload_seconds", time.monotonic() - t)
            log.info("[%s] output %s: %s -> %s via %s (%s)", job_id, render.name,
                     human_bytes(out_info.size_bytes), destination.get("uri") or destination.get("url")
                     or destination.get("path") or destination.get("type"), outcome.pipeline.value, outcome.encoder)
            outputs.append(_output_record(render, media, out_info, outcome, destination, render_warnings))
            try:
                os.remove(out_path)
            except OSError:
                pass

        result: dict[str, Any] = {
            "status": "success",
            "job_id": job_id,
            "input": {"source": source.display, **media.summary()},
            "outputs": outputs,
            "timings": timings.to_dict(),
            "worker": _worker_info(settings, caps),
        }
        if job_warnings:
            result["warnings"] = job_warnings
        log.info("[%s] success in %.1fs", job_id, result["timings"]["total_seconds"])
        return result

    except VideoServiceError as exc:
        log.error("[%s] %s: %s (hint: %s)", job_id, exc.code, exc.message, exc.hint)
        if exc.details:
            log.error("[%s] details: %s", job_id, exc.details)
        return _error_response(job_id, exc.to_dict(), refresh_worker=exc.refresh_worker,
                               timings=timings, settings=settings, caps=caps)
    except Exception as exc:  # noqa: BLE001 - last line of defence, must never fail silently
        log.exception("[%s] unexpected error", job_id)
        payload = {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}", "retryable": False,
                   "details": {"traceback": traceback.format_exc()[-4000:]},
                   "hint": "This is a bug in the service; please report the traceback."}
        return _error_response(job_id, payload, refresh_worker=False, timings=timings, settings=settings, caps=caps)
    finally:
        if workspace and os.path.isdir(workspace):
            if settings is not None and settings.keep_work_dir:
                log.info("[%s] KEEP_WORK_DIR set, leaving %s", job_id, workspace)
            else:
                shutil.rmtree(workspace, ignore_errors=True)


def _error_response(job_id: str, error: dict[str, Any], *, refresh_worker: bool, timings: Timings,
                    settings: Settings | None, caps: Capabilities | None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "status": "error",
        "job_id": job_id,
        # RunPod pops this key and marks the job FAILED; keep it a readable one-liner.
        "error": f"{error['code']}: {error['message']}",
        "error_detail": error,
        "timings": timings.to_dict(),
    }
    if settings is not None:
        response["worker"] = _worker_info(settings, caps)
    if refresh_worker:
        response["refresh_worker"] = True
    return response
