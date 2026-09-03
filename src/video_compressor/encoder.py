"""Run FFmpeg, watch progress, classify failures, fall back between pipelines."""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .capabilities import Capabilities
from .config import Settings
from .errors import EncodeError, EncodeTimeoutError, GpuError, VideoServiceError
from .ffmpeg_cmd import BuiltCommand, Pipeline, RenderSpec, build_command, compute_geometry, gpu_decode_eligible
from .probe import MediaInfo

log = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]


# --------------------------------------------------------------------------
# Failure classification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FailureClass:
    code: str
    hint: str
    gpu_related: bool = False   # NVENC/driver problem: skip further NVENC attempts, recycle worker
    fatal: bool = False         # No pipeline can succeed (bad input, disk full, ...)
    retryable: bool = False


_PATTERNS: list[tuple[re.Pattern, FailureClass]] = [
    (re.compile(r"Cannot load libnvidia-encode|libnvidia-encode\.so|Cannot load libcuda|libcuda\.so\.1", re.I),
     FailureClass("GPU_LIBRARIES_NOT_VISIBLE",
                  "The NVIDIA driver libraries are not visible inside the container. Make sure the worker "
                  "runs on a GPU and the image sets NVIDIA_DRIVER_CAPABILITIES=compute,utility,video "
                  "(the Dockerfile does). On a Pod, check `nvidia-smi` works.", gpu_related=True)),
    (re.compile(r"Driver does not support the required nvenc API version", re.I),
     FailureClass("NVENC_DRIVER_TOO_OLD",
                  "The host NVIDIA driver is older than the NVENC API this FFmpeg build was compiled against "
                  "(FFmpeg 8.x needs driver >= 570). In the RunPod endpoint settings restrict 'Allowed CUDA "
                  "versions' to 12.8+, or rebuild the image with an older FFmpeg via --build-arg FFMPEG_URL.",
                  gpu_related=True)),
    (re.compile(r"No capable devices found|No NVENC capable devices|OpenEncodeSessionEx failed: unsupported device"
                r"|Codec not supported|is not supported on this GPU|Cannot initialize encoder", re.I),
     FailureClass("NVENC_NOT_SUPPORTED_ON_GPU",
                  "This GPU has no NVENC engine for the requested codec. A100/H100 have no NVENC at all; "
                  "av1_nvenc needs an Ada-generation GPU (RTX 40xx, L4, L40). Choose RTX A5000/A6000/A40/4090 "
                  "or use codec h264/hevc.", gpu_related=True)),
    (re.compile(r"OpenEncodeSessionEx failed: out of memory|too many concurrent sessions|out of memory \(10\)", re.I),
     FailureClass("NVENC_SESSION_LIMIT",
                  "The NVENC session limit of this GPU was hit (consumer GPUs allow only a few concurrent "
                  "encodes). Run one job per worker and retry.", gpu_related=True, retryable=True)),
    (re.compile(r"InitializeEncoder failed: invalid param|InitializeEncoder failed: unsupported param"
                r"|Invalid Level|Unsupported (profile|level|pixel format)|error setting", re.I),
     FailureClass("NVENC_INVALID_PARAMETERS",
                  "NVENC rejected the encoding parameters for this GPU (e.g. b_ref_mode/bframes on an older "
                  "GPU, 4:4:4 or 10-bit with h264). Simplify video.* options (bframes=0, b_ref_mode=disabled, "
                  "pixel_format=yuv420p) or remove extra_args.")),
    (re.compile(r"Impossible to convert between the formats|Failed setup for format cuda|hwaccel initialisation"
                r"|Error while decoding stream|No decoder surfaces left|Failed to initialise CUDA|cuvid|nvdec|"
                r"CUDA_ERROR|Auto hwaccel|Device creation failed|hw_frames_ctx", re.I),
     FailureClass("HW_DECODE_FAILED",
                  "GPU decoding/filtering failed for this source; the service retries with CPU decoding "
                  "automatically unless options.hw_decode='on'.")),
    (re.compile(r"Invalid data found when processing input|moov atom not found|Error demuxing|could not find codec"
                r" parameters|Header missing|Unable to find a suitable output|does not contain any stream", re.I),
     FailureClass("INPUT_INVALID",
                  "FFmpeg could not read the input. The file is corrupt, truncated, not a video, or the "
                  "download returned an error page instead of the video.", fatal=True)),
    (re.compile(r"No space left on device", re.I),
     FailureClass("DISK_FULL",
                  "The worker ran out of disk. Increase the container disk size of the RunPod endpoint "
                  "(2-3x the input size is a safe rule) or use a network volume as WORK_DIR.", fatal=True)),
    (re.compile(r"Unknown encoder", re.I),
     FailureClass("ENCODER_MISSING", "This FFmpeg build lacks the requested encoder. Rebuild the image "
                  "with an FFmpeg that includes it.")),
    (re.compile(r"Could not find tag for codec|Could not write header|not supported in this container|"
                r"Unsupported codec id|Only VP8 or VP9 or AV1", re.I),
     FailureClass("CONTAINER_CODEC_MISMATCH", "The chosen container cannot store the chosen codec/stream. "
                  "Use container='mkv' or a different codec.", fatal=True)),
    (re.compile(r"Permission denied", re.I),
     FailureClass("PERMISSION_DENIED", "FFmpeg could not read/write a file. Check WORK_DIR permissions.",
                  fatal=True)),
]


def classify_failure(stderr_tail: str, returncode: int | None) -> FailureClass:
    if returncode is not None and returncode < 0 and -returncode == signal.SIGKILL:
        return FailureClass("PROCESS_KILLED",
                            "FFmpeg was killed (SIGKILL), usually by the OOM killer. Use a worker with more "
                            "RAM/VRAM or reduce lookahead/bframes.", fatal=True)
    for pattern, klass in _PATTERNS:
        if pattern.search(stderr_tail):
            return klass
    return FailureClass("ENCODE_FAILED", "See details.stderr_tail for the FFmpeg error output.")


# --------------------------------------------------------------------------
# Running one FFmpeg process
# --------------------------------------------------------------------------
@dataclass
class FfmpegRun:
    returncode: int | None
    elapsed_seconds: float
    stderr_tail: str
    last_progress: dict = field(default_factory=dict)
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _parse_progress_value(key: str, value: str) -> object:
    if key in ("frame", "total_size", "out_time_us", "out_time_ms", "dup_frames", "drop_frames"):
        try:
            return int(value)
        except ValueError:
            return None
    if key == "fps":
        try:
            return float(value)
        except ValueError:
            return None
    return value


def run_ffmpeg(
    args: list[str],
    *,
    timeout_seconds: int = 0,
    expected_duration: float | None = None,
    progress_cb: ProgressCallback | None = None,
    progress_interval: float = 5.0,
    log_file: str | None = None,
    stderr_lines_kept: int = 300,
) -> FfmpegRun:
    """Run FFmpeg (with ``-progress pipe:1``) and collect progress + stderr tail."""
    log.info("Running: %s", " ".join(_quote(a) for a in args))
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", bufsize=1, start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise VideoServiceError(
            f"FFmpeg binary not found: {args[0]!r}.", code="FFMPEG_NOT_FOUND",
            hint="Install FFmpeg or set FFMPEG_BIN.",
        ) from exc

    tail: deque[str] = deque(maxlen=stderr_lines_kept)
    log_handle = open(log_file, "w", encoding="utf-8") if log_file else None

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip("\n")
            tail.append(line)
            if log_handle:
                log_handle.write(line + "\n")

    reader = threading.Thread(target=_drain_stderr, name="ffmpeg-stderr", daemon=True)
    reader.start()

    timed_out = False
    last_progress: dict = {}
    current: dict = {}
    last_emit = 0.0

    def _kill() -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    watchdog: threading.Timer | None = None
    if timeout_seconds > 0:
        def _on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            log.error("FFmpeg exceeded timeout of %ss, killing it.", timeout_seconds)
            _kill()
        watchdog = threading.Timer(timeout_seconds, _on_timeout)
        watchdog.daemon = True
        watchdog.start()

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key == "progress":
                current["elapsed_seconds"] = round(time.monotonic() - started, 2)
                out_time_us = current.get("out_time_us")
                if isinstance(out_time_us, int):
                    current["out_time_seconds"] = round(out_time_us / 1_000_000, 3)
                    if expected_duration and expected_duration > 0:
                        current["percent"] = round(min(100.0, 100.0 * out_time_us / 1_000_000 / expected_duration), 1)
                last_progress = dict(current)
                now = time.monotonic()
                if progress_cb and (value == "end" or now - last_emit >= progress_interval):
                    last_emit = now
                    try:
                        progress_cb(dict(last_progress))
                    except Exception:  # never let a progress hook break the encode
                        log.exception("progress callback failed")
                current = {}
            else:
                parsed = _parse_progress_value(key, value)
                if parsed is not None:
                    current[key] = parsed
        proc.wait()
    except BaseException:
        _kill()
        raise
    finally:
        if watchdog:
            watchdog.cancel()
        reader.join(timeout=10)
        if log_handle:
            log_handle.close()

    elapsed = time.monotonic() - started
    return FfmpegRun(
        returncode=proc.returncode, elapsed_seconds=elapsed, stderr_tail="\n".join(tail),
        last_progress=last_progress, timed_out=timed_out,
    )


def _quote(arg: str) -> str:
    return arg if re.match(r"^[\w./:=+,%-]+$", arg) else repr(arg)


# --------------------------------------------------------------------------
# Pipeline planning and fallback
# --------------------------------------------------------------------------
@dataclass
class AttemptRecord:
    pipeline: str
    encoder: str
    elapsed_seconds: float
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EncodeOutcome:
    pipeline: Pipeline
    encoder: str
    command: list[str]
    elapsed_seconds: float
    attempts: list[AttemptRecord]
    warnings: list[str]
    last_progress: dict
    geometry_out: tuple[int, int]


def plan_pipelines(settings: Settings, caps: Capabilities, render: RenderSpec, media: MediaInfo) -> tuple[list[Pipeline], list[str]]:
    """Decide which pipelines to try, in order, and why others were skipped."""
    notes: list[str] = []
    backend = settings.encoder_backend
    nvenc_ok = caps.nvenc_available
    want_nvenc = backend in ("nvenc", "auto")
    plan: list[Pipeline] = []

    if want_nvenc and not nvenc_ok:
        if backend == "nvenc" and not render.options.fallback_to_software:
            klass = classify_failure(caps.nvenc_error_log or caps.nvenc_error or "", None)
            code = klass.code if klass.code != "ENCODE_FAILED" else "NVENC_UNAVAILABLE"
            hint = klass.hint if klass.code != "ENCODE_FAILED" else (
                "Check the worker has an NVENC-capable GPU (not A100/H100) and a recent driver; "
                "set options.fallback_to_software=true to encode on the CPU instead.")
            raise GpuError(
                f"NVENC is not available on this worker (ENCODER_BACKEND=nvenc is strict): {caps.nvenc_error}",
                code=code, hint=hint,
                details={"nvenc_error": caps.nvenc_error, "nvenc_error_log": _tail(caps.nvenc_error_log or "", 15),
                         "gpu": caps.gpu_name, "driver_version": caps.driver_version,
                         "ffmpeg_version": caps.ffmpeg_version},
            )
        notes.append(f"NVENC unavailable ({caps.nvenc_error}); using software encoders.")
        want_nvenc = False

    if want_nvenc:
        geometry = compute_geometry(media, render.video, render.aspect)
        eligible, reason = gpu_decode_eligible(media, render, geometry)
        if render.options.hw_decode == "on":
            if not eligible:
                raise EncodeError(
                    f"options.hw_decode='on' but GPU decoding is impossible for this source: {reason}.",
                    code="HW_DECODE_NOT_POSSIBLE", hint="Set options.hw_decode to 'auto' or 'off'.",
                )
            plan.append(Pipeline.GPU_DECODE_NVENC)
        elif render.options.hw_decode == "auto":
            if eligible:
                plan.append(Pipeline.GPU_DECODE_NVENC)
            else:
                notes.append(f"GPU decode skipped: {reason}.")
            plan.append(Pipeline.CPU_DECODE_NVENC)
        else:
            plan.append(Pipeline.CPU_DECODE_NVENC)

    if backend == "software" or not want_nvenc or render.options.fallback_to_software:
        plan.append(Pipeline.SOFTWARE)
    return plan, notes


def _software_available(caps: Capabilities, render: RenderSpec) -> str | None:
    from .ffmpeg_cmd import SOFTWARE_ENCODER

    enc = SOFTWARE_ENCODER[render.video.codec]
    return None if caps.has_encoder(enc) else enc


def encode_with_fallback(
    settings: Settings,
    caps: Capabilities,
    render: RenderSpec,
    media: MediaInfo,
    input_path: str,
    output_path: str,
    *,
    progress_cb: ProgressCallback | None = None,
    log_dir: str | None = None,
) -> EncodeOutcome:
    plan, notes = plan_pipelines(settings, caps, render, media)
    warnings: list[str] = list(notes)
    attempts: list[AttemptRecord] = []
    timeout = render.options.timeout_seconds or settings.ffmpeg_timeout_seconds or 0
    expected = render.expected_duration or media.duration
    last_class: FailureClass | None = None
    last_run: FfmpegRun | None = None
    skip_nvenc = False

    for index, pipeline in enumerate(plan):
        if skip_nvenc and pipeline.uses_nvenc:
            attempts.append(AttemptRecord(pipeline.value, "-", 0.0, "SKIPPED", "skipped after a GPU-level failure"))
            continue
        if pipeline is Pipeline.SOFTWARE:
            missing = _software_available(caps, render)
            if missing:
                attempts.append(AttemptRecord(pipeline.value, missing, 0.0, "ENCODER_MISSING",
                                              f"{missing} not compiled into this FFmpeg"))
                continue

        built: BuiltCommand = build_command(settings, render, media, pipeline, input_path, output_path)
        warnings.extend(w for w in built.warnings if w not in warnings)
        if os.path.exists(output_path):
            os.remove(output_path)
        log_file = os.path.join(log_dir, f"ffmpeg_{render.name}_{index}_{pipeline.value}.log") if log_dir else None
        log.info("Encode attempt %d/%d for %s via %s (%s)", index + 1, len(plan), render.name, pipeline.value, built.encoder)

        run = run_ffmpeg(
            built.args, timeout_seconds=timeout, expected_duration=expected, progress_cb=progress_cb,
            progress_interval=settings.progress_interval_seconds, log_file=log_file,
        )
        last_run = run
        if run.ok and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            attempts.append(AttemptRecord(pipeline.value, built.encoder, round(run.elapsed_seconds, 2)))
            if index > 0:
                warnings.append(f"Fell back to pipeline '{pipeline.value}' after earlier attempt(s) failed.")
            return EncodeOutcome(
                pipeline=pipeline, encoder=built.encoder, command=built.args,
                elapsed_seconds=run.elapsed_seconds, attempts=attempts, warnings=warnings,
                last_progress=run.last_progress, geometry_out=(built.geometry.out_w, built.geometry.out_h),
            )

        if run.timed_out:
            raise EncodeTimeoutError(
                f"FFmpeg exceeded the timeout of {timeout}s while encoding {render.name!r} "
                f"(pipeline {pipeline.value}).",
                details={"timeout_seconds": timeout, "last_progress": run.last_progress,
                         "stderr_tail": _tail(run.stderr_tail), "attempts": [a.to_dict() for a in attempts]},
                hint="Raise options.timeout_seconds / FFMPEG_TIMEOUT_SECONDS and the RunPod job "
                     "executionTimeout, or use a faster preset.",
            )

        klass = classify_failure(run.stderr_tail, run.returncode)
        if klass.code == "HW_DECODE_FAILED" and pipeline is not Pipeline.GPU_DECODE_NVENC:
            klass = FailureClass("ENCODE_FAILED", "See details.stderr_tail for the FFmpeg error output.")
        last_class = klass
        message = _last_error_line(run.stderr_tail) or f"ffmpeg exited with code {run.returncode}"
        attempts.append(AttemptRecord(pipeline.value, built.encoder, round(run.elapsed_seconds, 2), klass.code, message))
        log.warning("Attempt via %s failed (%s): %s", pipeline.value, klass.code, message)
        if klass.gpu_related:
            skip_nvenc = True
        if klass.fatal:
            break

    # Every attempt failed.
    assert last_run is not None or attempts, "no pipeline was attempted"
    klass = last_class or FailureClass("ENCODE_FAILED", "No encoding pipeline could run.")
    details = {
        "attempts": [a.to_dict() for a in attempts],
        "stderr_tail": _tail(last_run.stderr_tail) if last_run else "",
        "returncode": last_run.returncode if last_run else None,
        "gpu": caps.gpu_name, "driver_version": caps.driver_version, "ffmpeg_version": caps.ffmpeg_version,
    }
    summary = attempts[-1].error_message if attempts else "no attempt made"
    if klass.gpu_related:
        raise GpuError(f"NVENC encoding failed ({klass.code}): {summary}", code=klass.code,
                       hint=klass.hint, details=details, retryable=klass.retryable)
    raise EncodeError(f"Encoding failed ({klass.code}): {summary}", code=klass.code,
                      hint=klass.hint, details=details, retryable=klass.retryable)


def _tail(text: str, lines: int = 40, max_chars: int = 6000) -> str:
    tail = "\n".join(text.splitlines()[-lines:])
    return tail[-max_chars:]


def _last_error_line(stderr: str) -> str | None:
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if not ln.lower().startswith(("conversion failed", "exiting normally")) and not ln.startswith("["):
            return ln[:300]
    return lines[-1][:300] if lines else None
