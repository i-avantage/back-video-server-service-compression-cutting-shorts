# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# GPU video compression worker for RunPod Serverless (FFmpeg + NVENC, Python).
#
#   docker build -t <registry>/<name>:<tag> .
#
# Build arguments:
#   CUDA_IMAGE    Official NVIDIA CUDA image. CUDA 12.8 <-> driver 570+, which is
#                 also what the bundled FFmpeg 8.x NVENC build requires.
#   FFMPEG_URL    Static FFmpeg build WITH NVENC (BtbN GPL build). The build fails
#                 loudly if the downloaded FFmpeg lacks h264_nvenc/hevc_nvenc/av1_nvenc.
#   FFMPEG_SHA256 Optional checksum to pin the FFmpeg tarball.
# ---------------------------------------------------------------------------
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
ARG FFMPEG_SHA256=

# Expose the driver's video (NVENC/NVDEC) libraries inside the container.
# Without "video" FFmpeg fails with "Cannot load libnvidia-encode.so.1".
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:/opt/ffmpeg/bin:${PATH} \
    FFMPEG_BIN=/opt/ffmpeg/bin/ffmpeg \
    FFPROBE_BIN=/opt/ffmpeg/bin/ffprobe \
    WORK_DIR=/tmp/video-jobs \
    ENCODER_BACKEND=nvenc \
    PYTHONPATH=/app/src

# Minimal OS deps: TLS roots + curl/xz to fetch FFmpeg, Python 3.12 (Ubuntu 24.04).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl xz-utils python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

# FFmpeg with NVENC. Verified at build time (encoder list does not need a GPU).
RUN set -eux; \
    curl -fsSL --retry 5 --retry-delay 3 -o /tmp/ffmpeg.tar.xz "${FFMPEG_URL}"; \
    if [ -n "${FFMPEG_SHA256}" ]; then echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c -; fi; \
    mkdir -p /opt/ffmpeg; \
    tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1; \
    rm -rf /tmp/ffmpeg.tar.xz /opt/ffmpeg/doc /opt/ffmpeg/man /opt/ffmpeg/presets /opt/ffmpeg/bin/ffplay; \
    /opt/ffmpeg/bin/ffmpeg -version | head -n 1; \
    /opt/ffmpeg/bin/ffmpeg -hide_banner -encoders | grep -E "h264_nvenc|hevc_nvenc|av1_nvenc"; \
    /opt/ffmpeg/bin/ffmpeg -hide_banner -encoders | grep -q h264_nvenc || { echo "ERROR: FFmpeg build has no NVENC encoders" >&2; exit 1; }; \
    /opt/ffmpeg/bin/ffmpeg -hide_banner -filters | grep -q scale_cuda || { echo "ERROR: FFmpeg build has no scale_cuda filter" >&2; exit 1; }; \
    /opt/ffmpeg/bin/ffmpeg -hide_banner -hwaccels | grep -q cuda || { echo "ERROR: FFmpeg build has no cuda hwaccel" >&2; exit 1; }

# Python virtualenv with the RunPod SDK and the worker's dependencies.
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN /opt/venv/bin/pip install -r /app/requirements.txt

COPY src /app/src
COPY handler.py /app/handler.py
WORKDIR /app

# Fail the build if the code does not even import.
RUN python -c "import runpod, boto3, pydantic, video_compressor; print('imports ok, runpod', runpod.__version__)" \
 && mkdir -p ${WORK_DIR}

CMD ["python", "-u", "/app/handler.py"]
