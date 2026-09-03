"""Detect what this worker can do: FFmpeg build, NVENC availability, GPU."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field

from .config import Settings
from .errors import VideoServiceError

log = logging.getLogger(__name__)

NVENC_ENCODERS = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
SOFTWARE_ENCODERS = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}


@dataclass
class Capabilities:
    ffmpeg_version: str = "unknown"
    encoders: set[str] = field(default_factory=set)
    nvenc_available: bool = False
    nvenc_error: str | None = None
    nvenc_error_log: str | None = None
    gpu_name: str | None = None
    driver_version: str | None = None

    def has_encoder(self, name: str) -> bool:
        return name in self.encoders

    def summary(self) -> dict:
        return {
            "ffmpeg_version": self.ffmpeg_version,
            "nvenc_available": self.nvenc_available,
            "nvenc_error": self.nvenc_error,
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "nvenc_encoders_compiled": sorted(e for e in self.encoders if e.endswith("_nvenc")),
        }


_lock = threading.Lock()
_cache: dict[str, Capabilities] = {}


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _ffmpeg_version(ffmpeg_bin: str) -> str:
    try:
        proc = _run([ffmpeg_bin, "-version"], timeout=30)
    except FileNotFoundError as exc:
        raise VideoServiceError(
            f"FFmpeg binary not found: {ffmpeg_bin!r}.",
            code="FFMPEG_NOT_FOUND",
            hint="Install FFmpeg (the Docker image ships one) or set FFMPEG_BIN to its path.",
        ) from exc
    first = (proc.stdout or proc.stderr or "").splitlines()[:1]
    match = re.search(r"ffmpeg version (\S+)", first[0]) if first else None
    return match.group(1) if match else "unknown"


def _list_encoders(ffmpeg_bin: str) -> set[str]:
    proc = _run([ffmpeg_bin, "-hide_banner", "-encoders"], timeout=30)
    names: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        match = re.match(r"^\s*[VAS][F.][S.][X.][B.][D.]\s+(\S+)", line)
        if match:
            names.add(match.group(1))
    return names


def _nvidia_smi() -> tuple[str | None, str | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        proc = _run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None, None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    return (parts[0] or None, parts[1] if len(parts) > 1 else None)


def nvenc_smoke_test(ffmpeg_bin: str, encoder: str = "h264_nvenc", timeout: int = 90) -> tuple[bool, str | None]:
    """Encode a handful of synthetic frames with NVENC. Returns (ok, error_text)."""
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=0.2",
        "-frames:v", "3", "-c:v", encoder, "-preset", "p1", "-f", "null", "-",
    ]
    try:
        proc = _run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"NVENC smoke test timed out after {timeout}s"
    if proc.returncode == 0:
        return True, None
    stderr = (proc.stderr or "").strip()
    return False, stderr[-3000:] or f"ffmpeg exited with {proc.returncode}"


_INTERESTING = re.compile(r"cannot load|driver|no capable|not supported|out of memory|failed|error", re.I)


def summarise_ffmpeg_error(stderr: str) -> str:
    """Pick the single most informative line out of FFmpeg's stderr."""
    lines = [re.sub(r"^\[[^\]]*\]\s*", "", ln).strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in lines:
        if _INTERESTING.search(ln) and "Terminating thread" not in ln and "Task finished" not in ln:
            return ln[:300]
    return lines[0][:300] if lines else "unknown error"


def detect_capabilities(settings: Settings, *, force: bool = False) -> Capabilities:
    """Detect once per process (cached per FFmpeg binary)."""
    with _lock:
        cached = _cache.get(settings.ffmpeg_bin)
        if cached is not None and not force:
            return cached
        caps = Capabilities(ffmpeg_version=_ffmpeg_version(settings.ffmpeg_bin))
        caps.encoders = _list_encoders(settings.ffmpeg_bin)
        caps.gpu_name, caps.driver_version = _nvidia_smi()
        if "h264_nvenc" not in caps.encoders:
            caps.nvenc_error = "This FFmpeg build has no NVENC encoders compiled in (h264_nvenc missing)."
        elif settings.encoder_backend == "software":
            caps.nvenc_error = "ENCODER_BACKEND=software: NVENC not probed."
        else:
            ok, err = nvenc_smoke_test(settings.ffmpeg_bin)
            caps.nvenc_available = ok
            if not ok and err:
                caps.nvenc_error_log = err
                caps.nvenc_error = summarise_ffmpeg_error(err)
        log.info("Capabilities: %s", caps.summary())
        _cache[settings.ffmpeg_bin] = caps
        return caps


def reset_cache() -> None:
    with _lock:
        _cache.clear()


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import json

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(detect_capabilities(Settings.from_env(), force=True).summary(), indent=2))
