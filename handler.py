"""RunPod Serverless entrypoint.

    python handler.py                       # inside the worker (RunPod calls this)
    python handler.py --test_input '{"input": {...}}'   # one local run via the SDK
    python handler.py --rp_serve_api        # local HTTP API on :8000 for manual testing

The actual work happens in ``video_compressor.service.process_job``.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import runpod  # noqa: E402

from video_compressor.capabilities import detect_capabilities  # noqa: E402
from video_compressor.config import Settings  # noqa: E402
from video_compressor.service import configure_logging, process_job  # noqa: E402

log = logging.getLogger("handler")


def handler(job: dict) -> dict:
    """Process one job. Always returns a dict (success or structured error)."""
    return process_job(job)


def warm_up() -> None:
    """Probe FFmpeg/NVENC once at start so problems show in the worker logs immediately."""
    try:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        log.info("Settings: %s", settings.redacted())
        caps = detect_capabilities(settings)
        if settings.encoder_backend != "software" and not caps.nvenc_available:
            log.error("NVENC is NOT available on this worker: %s", caps.nvenc_error)
        else:
            log.info("Worker ready (backend=%s, gpu=%s, ffmpeg=%s)",
                     settings.encoder_backend, caps.gpu_name, caps.ffmpeg_version)
    except Exception:  # noqa: BLE001 - never prevent the worker from starting; jobs will report the error
        log.exception("Warm-up failed")


if __name__ == "__main__":
    warm_up()
    runpod.serverless.start({"handler": handler})
