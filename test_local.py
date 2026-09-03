#!/usr/bin/env python3
"""Run the handler locally, without RunPod, against a small sample video.

Examples
--------
    python test_local.py                      # generate a sample, compress it on CPU or GPU (auto)
    python test_local.py --backend nvenc      # require NVENC (fails clearly if there is no GPU)
    python test_local.py --input my.mp4 --codec hevc --cq 26 --max-height 720
    python test_local.py --shorts             # cut two 9:16 clips from the sample
    python test_local.py --payload examples/compress_to_s3.json   # run any payload as-is

Outputs land in ./local_output by default. Exit code is 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def make_sample(path: str, ffmpeg: str, seconds: int = 8) -> None:
    """Generate a synthetic 720p test clip with audio (needs an FFmpeg with libx264)."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"Could not generate the sample video with {ffmpeg}:\n{proc.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="Path or URL of a video (default: generate a synthetic sample)")
    parser.add_argument("--output-dir", default="local_output")
    parser.add_argument("--backend", choices=["auto", "nvenc", "software"], default=os.environ.get("ENCODER_BACKEND", "auto"))
    parser.add_argument("--codec", default="h264", choices=["h264", "hevc", "av1"])
    parser.add_argument("--cq", type=int, default=23)
    parser.add_argument("--preset", default="p5")
    parser.add_argument("--max-height", type=int, default=None)
    parser.add_argument("--shorts", action="store_true", help="Produce two 9:16 clips instead of one file")
    parser.add_argument("--payload", help="JSON file with a full job payload ({'input': {...}}) to run as-is")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep FFmpeg logs / temp files")
    args = parser.parse_args()

    os.environ["ENCODER_BACKEND"] = args.backend
    os.environ.setdefault("WORK_DIR", os.path.abspath(os.path.join(args.output_dir, ".work")))
    if args.keep_work_dir:
        os.environ["KEEP_WORK_DIR"] = "true"

    from video_compressor.config import Settings
    from video_compressor.service import process_job

    settings = Settings.from_env()
    if shutil.which(settings.ffmpeg_bin) is None and not os.path.isfile(settings.ffmpeg_bin):
        sys.exit(f"FFmpeg not found ({settings.ffmpeg_bin}). Install it or set FFMPEG_BIN=/path/to/ffmpeg")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.payload:
        with open(args.payload, encoding="utf-8") as fh:
            job = json.load(fh)
        job.setdefault("id", f"local-{int(time.time())}")
    else:
        source = args.input
        if not source:
            source = os.path.join(out_dir, "sample_input.mp4")
            print(f"Generating sample video at {source} ...")
            make_sample(source, settings.ffmpeg_bin)
        elif not source.startswith(("http://", "https://", "s3://")):
            source = os.path.abspath(source)

        video = {"codec": args.codec, "cq": args.cq, "preset": args.preset}
        if args.max_height:
            video["max_height"] = args.max_height
        payload = {
            "input_url": source,
            "video": video,
            "audio": {"codec": "aac", "bitrate": "128k"},
            "output": {"type": "local", "path": out_dir, "overwrite": True},
            "options": {"fallback_to_software": args.backend != "nvenc"},
        }
        if args.shorts:
            payload["clips"] = [
                {"name": "short_1", "trim": {"start": 0, "duration": 3}, "aspect": {"ratio": "9:16", "mode": "crop"}},
                {"name": "short_2", "trim": {"start": "00:00:03", "end": "00:00:06"},
                 "aspect": {"ratio": "9:16", "mode": "pad", "pad_color": "black"}},
            ]
        job = {"id": f"local-{int(time.time())}", "input": payload}

    print("Job payload:")
    print(json.dumps(job, indent=2))
    print("\nRunning handler ...\n")

    def progress(p: dict) -> None:
        print("  progress:", json.dumps(p))

    result = process_job(job, progress_hook=progress)
    print("\nResult:")
    print(json.dumps(result, indent=2))
    if result.get("status") != "success":
        print("\nFAILED:", result.get("error"), file=sys.stderr)
        return 1
    for out in result["outputs"]:
        dest = out["destination"]
        print(f"\nOK: {out['name']} -> {dest.get('path') or dest.get('uri') or dest.get('type')} "
              f"({out['size_bytes']} bytes, {out['width']}x{out['height']}, {out['pipeline']}/{out['encoder']}, "
              f"reduction {out['size_reduction_percent']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
