# GPU video compression, cutting & shorts service for RunPod Serverless

A containerised video transcoding service built for **RunPod Serverless**. It downloads a source
video, encodes it with **FFmpeg + NVIDIA NVENC** (h264 / hevc / av1 hardware encoding), optionally
**cuts segments** and **reframes to 9:16 shorts**, verifies the result with ffprobe, uploads it to
S3-compatible storage and returns a structured JSON result. Every failure is returned as an explicit,
machine-readable error with a hint. Nothing fails silently.

- **Stack**: Python 3.11+ orchestration, FFmpeg CLI via `subprocess`, RunPod Python SDK. No ML frameworks.
- **GPU**: any NVENC-capable NVIDIA GPU (RTX A5000 / A6000 / A40 / RTX 4090 / L4 / L40). Not A100 / H100 (they have no NVENC).
- **Modes**: RunPod Serverless endpoint (primary), RunPod Pod (debugging), or a laptop/CI with CPU encoders (testing).

---

## Table of contents

1. [How it works](#how-it-works)
2. [Quick start on your machine](#quick-start-on-your-machine)
3. [Build the Docker image](#build-the-docker-image)
4. [Push the image to a registry](#push-the-image-to-a-registry)
5. [Deploy on RunPod](#deploy-on-runpod)
6. [Storage: inputs and outputs](#storage-inputs-and-outputs)
7. [Environment variables](#environment-variables)
8. [Job input reference](#job-input-reference)
9. [Response format](#response-format)
10. [Calling the deployed endpoint](#calling-the-deployed-endpoint)
11. [Timeouts and long videos](#timeouts-and-long-videos)
12. [Cost control](#cost-control)
13. [Troubleshooting](#troubleshooting)
14. [Development and tests](#development-and-tests)
15. [Design decisions and assumptions](#design-decisions-and-assumptions)

---

## How it works

```
job payload ──► validate (strict schema) ──► download (https / s3://) ──► ffprobe
      ──► FFmpeg encode with fallback chain ──► ffprobe verification ──► upload ──► JSON result
```

Encoding pipelines, tried in order until one succeeds:

| Pipeline            | Decode          | Filters (scale / crop / pad / fps) | Encode                        |
|---------------------|-----------------|------------------------------------|-------------------------------|
| `gpu_decode_nvenc`  | NVDEC (GPU)     | `scale_cuda` on the GPU            | NVENC                         |
| `cpu_decode_nvenc`  | CPU             | CPU                                | NVENC                         |
| `software`          | CPU             | CPU                                | libx264 / libx265 / SVT-AV1   |

The GPU-decode pipeline is only attempted when the source is NVDEC-decodable and no CPU-only
filter is needed (crop / pad / fps change / rotation metadata). The software pipeline runs only when
`ENCODER_BACKEND=software|auto` or when a job sets `options.fallback_to_software=true`. The result
always reports which pipeline and encoder actually produced the file.

Repository layout:

```
handler.py                 RunPod Serverless entrypoint (runpod.serverless.start)
test_local.py              Run the handler locally against a generated sample video
src/video_compressor/
  schema.py                Strict job input validation (pydantic)
  service.py               Orchestration: download -> probe -> encode -> verify -> deliver
  ffmpeg_cmd.py            FFmpeg command construction (NVENC / software, geometry)
  encoder.py               FFmpeg runner: progress, timeout, failure classification, fallback
  storage.py               https / s3 download, S3 / presigned PUT / local / base64 delivery
  probe.py                 ffprobe wrapper
  capabilities.py          FFmpeg + NVENC + GPU detection (run at worker start)
  config.py                Settings from environment variables
  errors.py                Structured error types
examples/                  Ready-to-send job payloads
scripts/run_remote.py      Submit a job to a RunPod endpoint and poll for the result
Dockerfile                 CUDA base + FFmpeg with NVENC (verified at build time) + Python venv
.github/workflows/         CI (lint + tests) and image publish to ghcr.io
```

---

## Quick start on your machine

Requires Python 3.11+ and an FFmpeg binary. Without an NVIDIA GPU the service encodes on the CPU
(same code path, same validation, same result shape) so the logic can be validated before deploying.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Generate an 8 s sample and compress it (CPU or GPU, whichever is available)
python test_local.py

# 2. Cut two 9:16 shorts from the sample
python test_local.py --shorts

# 3. Your own file, hevc, downscale to 720p, require NVENC (fails clearly without a GPU)
python test_local.py --input /path/to/video.mp4 --codec hevc --cq 26 --max-height 720 --backend nvenc

# 4. Run any payload as-is (output goes where the payload says)
python test_local.py --payload examples/local_test_payload.json
```

Outputs land in `./local_output/`. Use `FFMPEG_BIN=/path/to/ffmpeg FFPROBE_BIN=/path/to/ffprobe`
if FFmpeg is not on your `PATH`. A static FFmpeg build with NVENC for Linux is available from
https://github.com/BtbN/FFmpeg-Builds (the same one the Docker image uses).

The RunPod SDK's own local mode also works:

```bash
python handler.py --test_input '{"input": {"input_url": "/abs/path/video.mp4", "output": {"type": "local", "path": "/tmp/out.mp4"}, "options": {"fallback_to_software": true}}}'
python handler.py --rp_serve_api      # local HTTP API on http://localhost:8000 (POST /run, /runsync)
```

---

## Build the Docker image

```bash
docker build -t ghcr.io/<your-org>/video-compressor:latest .
```

What the build does (and checks):

- Base image `nvidia/cuda:12.8.1-base-ubuntu24.04` (official NVIDIA image, lean "base" flavour; CUDA 12.8
  pairs with NVIDIA driver 570+, which the bundled FFmpeg's NVENC build also requires).
- Downloads a static **FFmpeg 8.1 build with NVENC** from BtbN and **fails the build** if
  `h264_nvenc` / `hevc_nvenc` / `av1_nvenc`, `scale_cuda` or the `cuda` hwaccel are missing.
  This check does not need a GPU, so a bad FFmpeg build is caught at build time, not at run time.
- Sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`. Without `video`, the driver's
  `libnvidia-encode.so` is not mounted into the container and NVENC fails.
- Installs Python 3.12 (Ubuntu 24.04 system Python, satisfies the 3.11+ requirement) and the pinned
  dependencies from `requirements.txt` into a virtualenv. No PyTorch, no CUDA toolkit.

Build arguments (all optional):

| Argument         | Default                                                        | Purpose |
|------------------|----------------------------------------------------------------|---------|
| `CUDA_IMAGE`     | `nvidia/cuda:12.8.1-base-ubuntu24.04`                          | Any official `nvidia/cuda:*-base-ubuntu24.04` image |
| `FFMPEG_URL`     | BtbN `ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz`               | Another static FFmpeg build with NVENC |
| `FFMPEG_SHA256`  | empty                                                          | Pin the tarball checksum |

Driver compatibility note: FFmpeg 8.x is compiled against NVENC API 13 and needs **driver 570 or
newer** on the host. If a worker logs `Driver does not support the required nvenc API version`,
either restrict the endpoint to CUDA 12.8+ hosts (see deployment) or build with an FFmpeg 7.x
tarball from a dated BtbN "autobuild" release via `--build-arg FFMPEG_URL=...` (FFmpeg 7.x needs
driver 470+ but has no AV1 NVENC). Verify the exact asset URL on the BtbN releases page.

Test the image on a machine with an NVIDIA GPU and the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all ghcr.io/<your-org>/video-compressor:latest python -m video_compressor.capabilities
# expect: "nvenc_available": true, plus GPU name and driver version
```

---

## Push the image to a registry

**GitHub Container Registry (ghcr.io)**

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u <github-user> --password-stdin   # token with write:packages
docker push ghcr.io/<your-org>/video-compressor:latest
```

**Docker Hub**

```bash
docker login
docker tag ghcr.io/<your-org>/video-compressor:latest <dockerhub-user>/video-compressor:latest
docker push <dockerhub-user>/video-compressor:latest
```

**Automatically with GitHub Actions**: `.github/workflows/docker-publish.yml` builds and pushes
`ghcr.io/<owner>/<repo>` on every push to `main` (tag `main` + `sha-<short>`) and on `v*.*.*` tags
(`1.2.3`, `1.2`, `latest`). It uses the built-in `GITHUB_TOKEN`; nothing to configure. If the
package is private, give RunPod pull credentials (below). `.github/workflows/ci.yml` runs lint and
the CPU test-suite on every push.

---

## Deploy on RunPod

### Serverless endpoint (primary target)

1. RunPod console → **Serverless** → **New Endpoint** → choose "Import from Docker registry" /
   custom image and enter the image, e.g. `ghcr.io/<your-org>/video-compressor:latest`.
   For a **private** image, first add the registry login under **Settings → Container Registry
   Auth** and select it on the endpoint/template.
2. **GPU selection**: choose a 24 GB or 48 GB class that includes NVENC GPUs
   (RTX A5000, RTX 4090, L4, A40, RTX A6000, L40). Do **not** select A100 / H100: they have no
   NVENC hardware and every job will fail with `NVENC_NOT_SUPPORTED_ON_GPU`. AV1 encoding needs an
   Ada-generation GPU (RTX 40xx, L4, L40).
3. **Container disk**: set it to at least 2.5× your largest input file (input + output + temp).
   Files are streamed to `WORK_DIR` (default `/tmp/video-jobs`) on the container disk.
   A too-small disk produces `INSUFFICIENT_DISK_SPACE` / `DISK_FULL` errors, not crashes.
4. **Allowed CUDA versions** (advanced settings): select 12.8 and above so workers land on hosts
   with driver 570+ (required by the bundled FFmpeg). If your console shows different labels,
   check the RunPod documentation for the current name of this filter.
5. **Environment variables**: add the storage variables from the [table below](#environment-variables)
   in the endpoint's environment settings (or in the template). Use RunPod **Secrets** for keys
   where available rather than plain values.
6. **Scaling**: Max workers as needed (one job runs per worker at a time, which is right for NVENC),
   Active/min workers `0` for scale-to-zero, Idle timeout 5–60 s, FlashBoot on. Execution timeout:
   see [Timeouts](#timeouts-and-long-videos).
7. Deploy. On the **Logs** tab the worker prints its capabilities at start:
   `Worker ready (backend=nvenc, gpu=NVIDIA RTX A5000, ffmpeg=n8.1...)`.
   If it prints `NVENC is NOT available on this worker: ...`, fix that before sending jobs.

### Pod (interactive debugging)

Create a Pod from the same image (GPU Pod, any NVENC GPU) with a small volume, then in the web
terminal:

```bash
nvidia-smi                                   # GPU visible?
python -m video_compressor.capabilities      # NVENC test encode + FFmpeg version
python test_local.py --backend nvenc         # end-to-end run with a synthetic sample
python handler.py --rp_serve_api --rp_api_host 0.0.0.0    # local HTTP API on port 8000 (expose it on the Pod to try requests)
```

Stop or terminate the Pod when done (see [Cost control](#cost-control)).

---

## Storage: inputs and outputs

**Decision taken** (per the brief's default): inputs arrive as a **signed https URL** or an
**`s3://bucket/key` reference**; outputs are **uploaded to S3-compatible storage** and the job
returns a reference plus a pre-signed download URL. Large binaries are never put in the job payload.

Inputs (`input_url`):

| Form                          | Notes |
|-------------------------------|-------|
| `https://…` (signed URL)      | Streamed with retries; error pages (HTML/JSON/XML) and HTTP 4xx/5xx are reported explicitly. Add `input_headers` for auth headers. |
| `s3://bucket/key`             | Downloaded with the worker's S3 credentials (env). |
| `/runpod-volume/…` (absolute) | A file already inside the container (network volume, or local tests). Not copied. |

Outputs (`output.type`):

| Type            | What happens | When to use |
|-----------------|--------------|-------------|
| `s3` (default)  | `upload_file` to `bucket`/`key` with the S3 env credentials, returns `s3://` URI + pre-signed GET URL (`S3_PRESIGN_TTL`). `bucket` defaults to `S3_BUCKET`, `key` to `S3_OUTPUT_PREFIX<job id>/<file>`. Existing objects are not overwritten unless `overwrite: true`. | Production |
| `presigned_put` | HTTP PUT of the file to `url` (a pre-signed upload URL you generate). No credentials on the worker at all. | When you do not want storage secrets in RunPod |
| `local`         | Copy to `path` inside the container (e.g. a network volume mounted at `/runpod-volume`). | Network volumes, local tests |
| `base64`        | Bytes inline in the response, capped at `MAX_INLINE_OUTPUT_BYTES` (8 MB). | Tiny test files only |

S3-compatible backends that work with the same variables: AWS S3, Cloudflare R2, Backblaze B2,
MinIO, Wasabi, Scaleway, and RunPod's S3-compatible network-volume API (set `S3_ENDPOINT_URL` to the
datacenter endpoint, `S3_FORCE_PATH_STYLE=true`, and the access key pair generated in the RunPod
console; verify the endpoint hostname in the RunPod docs for your datacenter).

---

## Environment variables

Set these on the RunPod endpoint (Environment Variables) or template. None are hard-coded.

| Variable | Default | Purpose |
|----------|---------|---------|
| `S3_ENDPOINT_URL` | (AWS) | Endpoint for non-AWS S3 (`https://<account>.r2.cloudflarestorage.com`, MinIO, RunPod, …) |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | – | Credentials (falls back to `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) |
| `S3_SESSION_TOKEN` | – | Optional STS token |
| `S3_REGION` | – | Region (`AWS_REGION` also accepted); `auto` for R2 |
| `S3_BUCKET` | – | Default output bucket. Required unless every job gives an `output` |
| `S3_OUTPUT_PREFIX` | `compressed/` | Key prefix for generated output keys |
| `S3_PRESIGN_TTL` | `3600` | Lifetime (s) of the pre-signed download URL in the result; `0` disables |
| `S3_FORCE_PATH_STYLE` | `false` | `true` for MinIO / RunPod / some providers |
| `S3_PUBLIC_BASE_URL` | – | If the bucket is public, also return `public_url = <base>/<key>` |
| `ENCODER_BACKEND` | `nvenc` (image) | `nvenc` = strict, fail if NVENC is missing; `auto` = NVENC else CPU; `software` = CPU only |
| `FFMPEG_TIMEOUT_SECONDS` | `0` (none) | Kill FFmpeg after N seconds → `ENCODE_TIMEOUT`. Per-job `options.timeout_seconds` overrides |
| `MAX_INPUT_BYTES` | `20000000000` | Refuse inputs larger than this (20 GB) |
| `MAX_INLINE_OUTPUT_BYTES` | `8000000` | Cap for `output.type=base64` |
| `DOWNLOAD_TIMEOUT_SECONDS` / `DOWNLOAD_ATTEMPTS` | `60` / `3` | HTTP read timeout and whole-transfer retries |
| `WORK_DIR` | `/tmp/video-jobs` | Scratch directory (container disk or network volume) |
| `KEEP_WORK_DIR` | `false` | Keep temp files + FFmpeg logs after a job (debugging only) |
| `LOG_LEVEL` | `INFO` | `DEBUG` prints more |
| `FFMPEG_BIN` / `FFPROBE_BIN` | image paths | Override binaries (local development) |

`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` is baked into the image and must stay.

---

## Job input reference

Send `{"input": {...}}`. Unknown fields are rejected (a typo never silently changes the encode).
Complete examples live in [`examples/`](examples/).

```jsonc
{
  "input": {
    "input_url": "https://…signed…/source.mp4",   // or s3://bucket/key, or /runpod-volume/file.mp4
    "input_headers": {"Authorization": "Bearer …"}, // optional, for https inputs
    "name": "interview_1080p",                      // output base name (letters, digits, . _ -)
    "container": "mp4",                             // mp4 (default) | mov | mkv

    "video": {
      "codec": "h264",              // h264 (default) | hevc | av1   (aliases: h265, avc)
      "preset": "p5",               // p1 fastest … p7 best quality (default p5)
      "tune": "hq",                 // hq | ll | ull | lossless
      "rate_control": "cq",         // cq (constant quality, default) | vbr | cbr
      "cq": 23,                     // quality for cq: 18 visually lossless … 28 small (av1 up to 63)
      "bitrate": "5M",              // required for vbr/cbr (forbidden in cq mode)
      "max_bitrate": "8M",          // optional peak cap (also works with cq)
      "buffer_size": "16M",         // optional VBV buffer
      "max_width": 1920, "max_height": 1080,   // downscale to fit, never upscale
      "width": 1080, "height": 1920,           // or exact size (even numbers); one of the two computes the other
      "fps": 30,                    // frame-rate conversion
      "pixel_format": "yuv420p",    // yuv420p (default) | nv12 | p010le (10-bit, hevc/av1) | yuv444p
      "profile": "high",            // h264: baseline/main/high/high444p; hevc: main/main10/rext; av1: main
      "level": "4.1",
      "gop_size": 60, "bframes": 3, "b_ref_mode": "middle",
      "multipass": "qres",          // disabled | qres (default) | fullres
      "spatial_aq": true, "temporal_aq": true, "aq_strength": 8, "lookahead": 20,
      "extra_args": ["-refs", "4"]  // advanced: raw FFmpeg output options
    },

    "audio": { "codec": "aac", "bitrate": "128k", "channels": 2, "sample_rate": 48000 },  // codec: aac | copy | none

    "trim":   { "start": "00:01:30", "end": "00:04:00" },      // or {"start": 90, "duration": 150}
    "aspect": { "ratio": "9:16", "mode": "crop", "anchor": "center" },   // mode: crop | pad, pad_color
    "clips": [                                                  // several outputs from one download (shorts)
      { "name": "hook", "trim": { "start": 130, "duration": 45 } },
      { "name": "outro", "trim": { "start": "00:41:00", "duration": 30 }, "aspect": { "ratio": "9:16", "mode": "pad" },
        "output": { "type": "s3", "key": "shorts/outro.mp4" } }
    ],

    "output": { "type": "s3", "bucket": "my-bucket", "key": "outputs/interview_1080p.mp4",
                "presign_ttl": 86400, "overwrite": false },
    // "output": { "type": "presigned_put", "url": "https://…", "headers": {} }
    // "output": { "type": "local", "path": "/runpod-volume/out/" }
    // "output": { "type": "base64" }

    "options": {
      "hw_decode": "auto",            // auto | on | off  (GPU decoding)
      "fallback_to_software": false,  // retry on CPU if NVENC fails (slow, but never silent: reported in the result)
      "timeout_seconds": 3600,        // kill FFmpeg after this
      "verify_output": true,          // ffprobe the result: duration/resolution/audio must match
      "faststart": true,              // mp4/mov streaming-friendly
      "progress_updates": true,       // RunPod IN_PROGRESS updates with percent / fps / speed
      "keep_metadata": true
    }
  },
  "policy": { "executionTimeout": 3600000 }   // RunPod job-level timeout in ms (see Timeouts)
}
```

Rules enforced up front (returned as `INVALID_INPUT` with the field name):
`bitrate` required for `vbr`/`cbr` and forbidden for `cq`; `width/height` cannot be combined with
`max_width/max_height`; dimensions must be even; `trim` needs `end` **or** `duration`; `trim` and
`clips` are mutually exclusive; clip names unique; `av1` cannot go in `.mov`; `h264` cannot be 10-bit;
`audio.codec=copy` is refused when the source audio cannot be muxed into the target container.

With `clips`, the top-level `video`, `audio`, `aspect` and `output` apply to every clip; a clip may
override `aspect` and `output`. Clips without their own `output` get `<key>_<clip name>.<ext>`.

---

## Response format

**Success** (`status: COMPLETED` on RunPod, `output` below):

```json
{
  "status": "success",
  "job_id": "3f9c…",
  "input": { "source": "https://…/source.mp4?<redacted>", "container": "mov,mp4,m4a,3gp,3g2,mj2",
             "duration_seconds": 612.4, "size_bytes": 1834112233, "bit_rate": 23960000,
             "video_codec": "h264", "width": 3840, "height": 2160, "fps": 29.97, "audio_codec": "aac" },
  "outputs": [
    {
      "name": "interview_1080p", "filename": "interview_1080p.mp4", "container": "mp4",
      "video_codec": "h264", "encoder": "h264_nvenc", "pipeline": "gpu_decode_nvenc",
      "width": 1920, "height": 1080, "fps": 29.97, "duration_seconds": 612.4,
      "size_bytes": 231455121, "bit_rate": 3023000, "audio_codec": "aac",
      "source_equivalent_bytes": 1834112233, "compression_ratio": 7.92, "size_reduction_percent": 87.4,
      "encode_seconds": 98.2, "encode_speed_x": 6.24,
      "attempts": [ { "pipeline": "gpu_decode_nvenc", "encoder": "h264_nvenc", "elapsed_seconds": 98.2 } ],
      "warnings": [],
      "destination": { "type": "s3", "bucket": "my-bucket", "key": "outputs/interview_1080p.mp4",
                       "uri": "s3://my-bucket/outputs/interview_1080p.mp4",
                       "url": "https://…presigned…", "url_expires_in_seconds": 86400, "size_bytes": 231455121 }
    }
  ],
  "timings": { "download_seconds": 41.2, "probe_seconds": 0.3, "encode_seconds": 98.2,
               "verify_seconds": 0.2, "upload_seconds": 12.9, "total_seconds": 153.1 },
  "worker": { "service_version": "1.0.0", "encoder_backend": "nvenc", "gpu": "NVIDIA RTX A5000",
              "driver_version": "570.xx", "ffmpeg_version": "n8.1…", "nvenc_available": true }
}
```

**Error** (RunPod marks the job `FAILED`; `error` is the one-line summary, `output` keeps the details):

```json
{
  "status": "error",
  "job_id": "3f9c…",
  "error": "DOWNLOAD_FAILED: Download of https://…/source.mp4?<redacted> failed with HTTP 403.",
  "error_detail": {
    "code": "DOWNLOAD_FAILED", "message": "…", "retryable": false,
    "hint": "For 403/404 the signed URL is probably expired, wrong, or unsigned.",
    "details": { "status_code": 403, "body_snippet": "<Error><Code>AccessDenied</Code>…" }
  },
  "timings": { "total_seconds": 0.8 },
  "worker": { "…": "…" }
}
```

Error codes:

| Code | Meaning / what to do |
|------|----------------------|
| `INVALID_INPUT` | Payload rejected; `details.problems[]` lists each field. Also used for trim beyond duration, missing local file, incompatible `audio.codec=copy`. |
| `STORAGE_NOT_CONFIGURED` | No `output` in the job and no `S3_BUCKET`; or S3 credentials missing. |
| `DOWNLOAD_FAILED`, `INPUT_TOO_LARGE` | HTTP/S3 download problems (status code and body snippet included). |
| `INSUFFICIENT_DISK_SPACE`, `DISK_FULL` | Container disk too small for input + output. |
| `PROBE_FAILED`, `UNSUPPORTED_MEDIA`, `INPUT_INVALID` | Not a readable video (corrupt, truncated, HTML error page, no video stream). |
| `GPU_LIBRARIES_NOT_VISIBLE` | `libnvidia-encode` / `libcuda` not mounted: no GPU attached or `NVIDIA_DRIVER_CAPABILITIES` lacks `video`. Worker is recycled. |
| `NVENC_DRIVER_TOO_OLD` | Host driver older than the FFmpeg NVENC build needs. Restrict CUDA versions or rebuild with older FFmpeg. Worker is recycled. |
| `NVENC_NOT_SUPPORTED_ON_GPU` | GPU has no NVENC (A100/H100) or no AV1 encoder (pre-Ada). Change GPU class or codec. |
| `NVENC_SESSION_LIMIT` | Too many concurrent NVENC sessions on this GPU. Retryable. |
| `NVENC_INVALID_PARAMETERS` | GPU rejected the parameter set (e.g. `b_ref_mode` on old GPUs). Simplify `video.*`. |
| `NVENC_UNAVAILABLE`, `GPU_ERROR` | NVENC probe failed for another reason; details carry the FFmpeg log. |
| `HW_DECODE_NOT_POSSIBLE` | `options.hw_decode=on` but the source cannot be GPU-decoded. |
| `ENCODE_FAILED` (+ specific codes) | All pipelines failed; `details.attempts[]` and `details.stderr_tail` show why. |
| `ENCODE_TIMEOUT` | `options.timeout_seconds` / `FFMPEG_TIMEOUT_SECONDS` exceeded; `details.last_progress` shows how far it got. |
| `OUTPUT_VERIFICATION_FAILED` | The file exists but its duration/resolution/audio does not match expectations (would have been a silent corruption). |
| `UPLOAD_FAILED`, `OUTPUT_ALREADY_EXISTS`, `OUTPUT_TOO_LARGE_FOR_INLINE` | Delivery problems. |
| `INTERNAL_ERROR` | Unexpected exception; traceback in `details`. Please report. |

`refresh_worker: true` is set on GPU-level errors so RunPod replaces the worker after the job.

---

## Calling the deployed endpoint

Get your API key (RunPod → Settings → API Keys) and the endpoint ID (Serverless → endpoint page).

**Asynchronous (recommended): `/run` then poll `/status/<job id>`**

```bash
export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...

curl -s -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{
        "input": {
          "input_url": "https://my-bucket.s3.amazonaws.com/in/talk.mp4?X-Amz-Signature=...",
          "video": {"codec": "h264", "cq": 23, "max_height": 1080},
          "output": {"type": "s3", "key": "out/talk_1080p.mp4", "overwrite": true}
        },
        "policy": {"executionTimeout": 3600000}
      }'
# -> {"id": "8d1f...", "status": "IN_QUEUE"}

curl -s "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/8d1f..." -H "Authorization: Bearer $RUNPOD_API_KEY"
# IN_PROGRESS responses carry the progress payload: {"stage": "encoding", "percent": 42.5, "fps": 187, "speed": "6.2x", ...}
# COMPLETED responses carry the result JSON shown above in "output"
```

**Python helper** (submits, prints progress, prints the result, exit code 1 on failure):

```bash
python scripts/run_remote.py examples/compress_to_s3.json --timeout-minutes 60
```

**Synchronous `/runsync`** returns the result directly but only waits a limited time on the server
side (RunPod documents the limit); use it for short clips only. `POST /cancel/<job id>` cancels a
queued or running job; whether an encode already running is interrupted depends on the RunPod SDK
version, so a cancelled long encode may keep the worker busy until it finishes or hits its timeout.

Multiple shorts from one video are one job (one download, N encodes); see
[`examples/shorts_from_one_video.json`](examples/shorts_from_one_video.json).

---

## Timeouts and long videos

Three timeouts matter:

1. **RunPod execution timeout** (endpoint setting, default 600 s = 10 min). A job that exceeds it is
   killed by RunPod with status `TIMED_OUT` and **no result from the handler**. Override per job
   with `"policy": {"executionTimeout": <ms>}` in the request. **Set this for every long encode**;
   the default is not enough for feature-length or 4K sources. Rough budget on an RTX A5000:
   download time + (duration ÷ 5–10 for 1080p h264, ÷ 2–4 for 4K) + upload time.
2. **Service-level FFmpeg timeout** (`options.timeout_seconds` or `FFMPEG_TIMEOUT_SECONDS`), which
   returns a structured `ENCODE_TIMEOUT` with progress information. Set it a little below the
   RunPod execution timeout so you get a descriptive error rather than a bare `TIMED_OUT`.
3. **Download timeouts** (`DOWNLOAD_TIMEOUT_SECONDS`, per read; transfers are retried whole).

The worker sends `IN_PROGRESS` updates every ~5 s (download %, encode %, fps, speed, upload) so a
stalled job is visible from `/status`.

---

## Cost control

- **Serverless scale-to-zero**: with *Active (min) workers = 0*, workers only exist while jobs run;
  after the *Idle timeout* (seconds) an idle worker is released and billing stops. Nothing stays
  running between jobs. Keep min workers at 0 unless you need to avoid cold starts (each active
  worker is billed continuously). Cold start for this image is a few seconds (no ML frameworks).
- **Max workers** caps parallelism and therefore the maximum hourly spend.
- **Pods used for testing are billed until stopped**: use *Stop* (keeps the volume, billed for storage
  only) or *Terminate* (deletes everything) from the Pods page when you are done. A stopped Pod
  keeps costing disk; terminate what you do not need.
- Choose the cheapest NVENC GPU class that fits (24 GB class: RTX A5000 / 4090 / L4). A100/H100 are
  both more expensive and unusable here.
- Container disk and network volumes are billed by size; size them for the largest input, not more.
- `S3_PRESIGN_TTL` is only a link lifetime and has no cost impact; egress from storage may.

---

## Troubleshooting

| Symptom (worker log / error) | Cause | Fix |
|------------------------------|-------|-----|
| `NVENC is NOT available … Cannot load libnvidia-encode.so.1` | Driver video libs not mounted | Ensure the endpoint uses GPU workers; `NVIDIA_DRIVER_CAPABILITIES` must include `video` (image default) |
| `Driver does not support the required nvenc API version` | Host driver < 570 | Restrict allowed CUDA versions to 12.8+, or build with FFmpeg 7.x (`FFMPEG_URL`) |
| `No capable devices found` | GPU without NVENC or without AV1 | Select RTX A5000/A40/4090/L4/L40; use h264/hevc unless the GPU is Ada |
| `INSUFFICIENT_DISK_SPACE` / `DISK_FULL` | Container disk too small | Increase container disk (≥ 2.5× input) |
| `DOWNLOAD_FAILED` with HTTP 403/404 | Expired/incorrect signed URL | Regenerate the URL; check clock/TTL |
| `… returned 'text/html' instead of a video` | URL points to a login/error page | Fix the URL or add `input_headers` |
| `OUTPUT_VERIFICATION_FAILED` | Encode ended early / file truncated | Retry; try `options.hw_decode: "off"`; check `details.attempts` and worker logs (FFmpeg log tail) |
| Job `TIMED_OUT` with no output | RunPod execution timeout | Add `policy.executionTimeout` to the request and `options.timeout_seconds` |
| Slow encodes with `pipeline: cpu_decode_nvenc` | Source not NVDEC-decodable or crop/pad/fps used | Expected; GPU decode is used whenever possible |
| `pipeline: software` in production | NVENC failed and `fallback_to_software` was on, or `ENCODER_BACKEND=auto` | Inspect `attempts[]`; set `ENCODER_BACKEND=nvenc` for strict behaviour |

Set `KEEP_WORK_DIR=true` and `LOG_LEVEL=DEBUG` on a Pod to keep the full FFmpeg logs
(`<WORK_DIR>/<job>/ffmpeg_*.log`) and the temp files.

---

## Development and tests

```bash
pip install -r requirements-dev.txt
ruff check src tests handler.py test_local.py scripts
pytest -q                     # unit tests + CPU integration tests (need ffmpeg with libx264 on PATH or FFMPEG_BIN)
python test_local.py --shorts # end-to-end on the CPU
```

The tests cover: schema validation, time/bitrate parsing, geometry (crop/pad/fit), FFmpeg command
construction for all three pipelines, failure classification, the FFmpeg runner (progress parsing,
timeout kill), HTTP download edge cases (403, HTML page, size cap), output resolution, and real
end-to-end runs of the handler (compression, trim, clips, base64, structured errors, timeout).
NVENC itself can only be exercised on a GPU: run `python test_local.py --backend nvenc` on a Pod.

---

## Design decisions and assumptions

Decisions taken where the brief left room (flag anything you want changed):

- **Defaults**: `h264_nvenc`, preset `p5`, `tune hq`, constant quality `cq 23` (`-rc vbr -cq 23 -b:v 0`),
  2-pass quarter-res (`multipass qres`), spatial + temporal AQ, 20-frame lookahead, 3 B-frames with
  `b_ref_mode middle`; AAC 128 kbit/s; MP4 with faststart; resolution and frame-rate unchanged.
  `hevc` outputs get the `hvc1` tag for Apple players.
- **Containers**: mp4, mov, mkv. WebM is intentionally not offered (NVENC h264/hevc cannot be muxed
  into it).
- **Outputs never inline** except `base64` under a hard cap, for tests.
- **`overwrite` defaults to false** for explicit S3 keys/paths: a re-run never silently replaces a
  delivered file. Generated keys include the job id and cannot collide.
- **Strict NVENC in production** (`ENCODER_BACKEND=nvenc`): a missing GPU is an error, never a silent
  CPU encode. CPU fallback is opt-in per job and reported in `outputs[].pipeline`.
- **Cutting and shorts** (`trim`, `clips`, `aspect`) are included because they are plain FFmpeg
  operations on the same pipeline and the repository name calls for them. Cuts are frame-accurate
  (seek + re-encode), not keyframe-aligned copies.
- **Verification**: every output is ffprobed; duration (±2 % / 0.5 s), resolution and presence of audio
  must match, otherwise the job fails with `OUTPUT_VERIFICATION_FAILED` rather than delivering a
  truncated file.
- **One job per worker**: NVENC session limits and RunPod's model make this the right default; scale
  with workers, not with in-worker concurrency.
- **Non-goals kept out**: no AI processing (the package layout leaves room for a future
  `filters` step), no web UI, no multi-cloud abstraction.
