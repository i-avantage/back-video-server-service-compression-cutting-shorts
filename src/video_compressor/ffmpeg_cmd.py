"""Build FFmpeg command lines for a render.

Three pipelines exist, tried in order until one succeeds (see ``encoder``):

* ``gpu_decode_nvenc`` – NVDEC decode, ``scale_cuda`` resize, NVENC encode (fastest).
* ``cpu_decode_nvenc`` – CPU decode + CPU filters, NVENC encode (most compatible GPU path).
* ``software``         – CPU decode, libx264 / libx265 / SVT-AV1 encode (no GPU needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .config import Settings
from .probe import MediaInfo
from .schema import (
    CONTAINER_AUDIO_COPY_OK,
    AspectSpec,
    AudioSpec,
    OptionsSpec,
    OutputSpec,
    TrimSpec,
    VideoSpec,
)
from .units import even, format_seconds


class Pipeline(StrEnum):
    GPU_DECODE_NVENC = "gpu_decode_nvenc"
    CPU_DECODE_NVENC = "cpu_decode_nvenc"
    SOFTWARE = "software"

    @property
    def uses_nvenc(self) -> bool:
        return self is not Pipeline.SOFTWARE


NVENC_ENCODER = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
SOFTWARE_ENCODER = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}
# NVENC p1..p7 -> x264/x265 presets (SVT-AV1 uses numeric presets, mapped separately).
X26X_PRESET = {"p1": "ultrafast", "p2": "superfast", "p3": "veryfast", "p4": "faster",
               "p5": "medium", "p6": "slow", "p7": "slower"}
SVT_PRESET = {"p1": "12", "p2": "11", "p3": "10", "p4": "9", "p5": "8", "p6": "6", "p7": "4"}
# Formats scale_cuda can output, keyed by our target pixel format.
CUDA_FORMAT = {"yuv420p": "nv12", "nv12": "nv12", "p010le": "p010le", "yuv444p": "yuv444p"}
# Codecs NVDEC handles on every Turing+ GPU with the pixel formats it accepts.
NVDEC_CODECS = {
    "h264": {"yuv420p", "yuvj420p", "nv12"},
    "hevc": {"yuv420p", "yuvj420p", "nv12", "yuv420p10le", "p010le"},
    "vp9": {"yuv420p", "yuv420p10le"},
    "av1": {"yuv420p", "yuv420p10le"},
    "mpeg2video": {"yuv420p"},
    "mpeg4": {"yuv420p"},
    "vc1": {"yuv420p"},
    "vp8": {"yuv420p"},
}
MUXER = {"mp4": "mp4", "mov": "mov", "mkv": "matroska"}


@dataclass
class RenderSpec:
    """Everything needed to produce one output file."""

    name: str
    filename: str
    container: str
    video: VideoSpec
    audio: AudioSpec
    options: OptionsSpec
    output: OutputSpec
    trim: TrimSpec | None = None
    aspect: AspectSpec | None = None

    @property
    def expected_duration(self) -> float | None:
        if self.trim and self.trim.end_seconds is not None:
            return self.trim.duration_seconds
        return None


@dataclass
class Geometry:
    source_w: int
    source_h: int
    crop: tuple[int, int, int, int] | None = None  # w, h, x, y
    pad: tuple[int, int, int, int] | None = None   # w, h, x, y
    scale: tuple[int, int] | None = None
    out_w: int = 0
    out_h: int = 0

    @property
    def needs_cpu_filters(self) -> bool:
        return self.crop is not None or self.pad is not None


@dataclass
class BuiltCommand:
    pipeline: Pipeline
    encoder: str
    args: list[str]
    video_filter: str | None
    geometry: Geometry
    warnings: list[str] = field(default_factory=list)


def compute_geometry(media: MediaInfo, video: VideoSpec, aspect: AspectSpec | None) -> Geometry:
    assert media.video is not None
    src_w, src_h = media.video.display_width, media.video.display_height
    geo = Geometry(source_w=src_w, source_h=src_h)
    w, h = src_w, src_h

    if aspect is not None:
        target = aspect.ratio_w / aspect.ratio_h
        current = w / h
        if abs(current - target) > 1e-3:
            if aspect.mode == "crop":
                if current > target:  # too wide -> crop width
                    new_w = max(2, even(int(round(h * target))))
                    x = {"left": 0, "right": w - new_w}.get(aspect.anchor, (w - new_w) // 2)
                    geo.crop = (new_w, h, even(x), 0)
                    w = new_w
                else:  # too tall -> crop height
                    new_h = max(2, even(int(round(w / target))))
                    y = {"top": 0, "bottom": h - new_h}.get(aspect.anchor, (h - new_h) // 2)
                    geo.crop = (w, new_h, 0, even(y))
                    h = new_h
            else:  # pad
                if current > target:  # too wide -> add height
                    new_h = even(int(round(w / target)) + 1)
                    y = {"top": 0, "bottom": new_h - h}.get(aspect.anchor, (new_h - h) // 2)
                    geo.pad = (w, new_h, 0, even(y))
                    h = new_h
                else:
                    new_w = even(int(round(h * target)) + 1)
                    x = {"left": 0, "right": new_w - w}.get(aspect.anchor, (new_w - w) // 2)
                    geo.pad = (new_w, h, even(x), 0)
                    w = new_w

    out_w, out_h = w, h
    if video.width and video.height:
        out_w, out_h = video.width, video.height
    elif video.width:
        out_w, out_h = video.width, even(int(round(video.width * h / w)))
    elif video.height:
        out_w, out_h = even(int(round(video.height * w / h))), video.height
    elif video.max_width or video.max_height:
        factor = 1.0
        if video.max_width:
            factor = min(factor, video.max_width / w)
        if video.max_height:
            factor = min(factor, video.max_height / h)
        if factor < 1.0:
            out_w, out_h = even(int(round(w * factor))), even(int(round(h * factor)))
    out_w, out_h = max(2, even(out_w)), max(2, even(out_h))
    if (out_w, out_h) != (w, h):
        geo.scale = (out_w, out_h)
    geo.out_w, geo.out_h = out_w, out_h
    return geo


def gpu_decode_eligible(media: MediaInfo, render: RenderSpec, geometry: Geometry) -> tuple[bool, str]:
    """Can the whole decode+filter chain stay on the GPU?"""
    v = media.video
    assert v is not None
    if v.codec not in NVDEC_CODECS:
        return False, f"source codec {v.codec!r} is not NVDEC-decodable"
    if v.pix_fmt not in NVDEC_CODECS[v.codec]:
        return False, f"source pixel format {v.pix_fmt!r} is not supported by NVDEC for {v.codec}"
    if v.rotation:
        return False, "source has rotation metadata (auto-rotate needs CPU filters)"
    if geometry.needs_cpu_filters:
        return False, "crop/pad filters run on the CPU"
    if render.video.fps:
        return False, "fps conversion runs on the CPU"
    if render.video.effective_pixel_format not in CUDA_FORMAT:
        return False, "target pixel format not supported by scale_cuda"
    return True, "ok"


def _rate_control_args(video: VideoSpec, pipeline: Pipeline) -> list[str]:
    args: list[str] = []
    if pipeline.uses_nvenc:
        if video.tune == "lossless":
            return []
        if video.rate_control == "cq":
            args += ["-rc", "vbr", "-cq", str(video.cq), "-b:v", "0"]
            if video.max_bitrate_bps:
                args += ["-maxrate", str(video.max_bitrate_bps),
                         "-bufsize", str(video.buffer_size_bits or 2 * video.max_bitrate_bps)]
        elif video.rate_control == "vbr":
            maxrate = video.max_bitrate_bps or int(video.bitrate_bps * 1.5)
            args += ["-rc", "vbr", "-b:v", str(video.bitrate_bps), "-maxrate", str(maxrate),
                     "-bufsize", str(video.buffer_size_bits or 2 * maxrate)]
        else:  # cbr
            args += ["-rc", "cbr", "-b:v", str(video.bitrate_bps), "-maxrate", str(video.bitrate_bps),
                     "-bufsize", str(video.buffer_size_bits or 2 * video.bitrate_bps)]
        return args
    # Software encoders
    if video.rate_control == "cq":
        crf = video.cq if video.tune != "lossless" else 0
        args += ["-crf", str(min(crf, 63 if video.codec == "av1" else 51))]
        if video.max_bitrate_bps and video.codec != "av1":
            args += ["-maxrate", str(video.max_bitrate_bps),
                     "-bufsize", str(video.buffer_size_bits or 2 * video.max_bitrate_bps)]
    elif video.rate_control == "vbr":
        maxrate = video.max_bitrate_bps or int(video.bitrate_bps * 1.5)
        args += ["-b:v", str(video.bitrate_bps), "-maxrate", str(maxrate),
                 "-bufsize", str(video.buffer_size_bits or 2 * maxrate)]
    else:
        args += ["-b:v", str(video.bitrate_bps), "-minrate", str(video.bitrate_bps),
                 "-maxrate", str(video.bitrate_bps), "-bufsize", str(video.buffer_size_bits or 2 * video.bitrate_bps)]
    return args


def _nvenc_video_args(video: VideoSpec) -> list[str]:
    args = ["-preset", video.preset, "-tune", video.tune, "-profile:v", video.effective_profile]
    args += _rate_control_args(video, Pipeline.CPU_DECODE_NVENC)
    if video.multipass != "disabled":
        args += ["-multipass", video.multipass]
    args += ["-spatial-aq", "1" if video.spatial_aq else "0"]
    if video.spatial_aq:
        args += ["-aq-strength", str(video.aq_strength)]
    args += ["-temporal-aq", "1" if video.temporal_aq else "0"]
    if video.lookahead:
        args += ["-rc-lookahead", str(video.lookahead)]
    if video.codec in ("h264", "hevc"):
        bframes = 3 if video.bframes is None else video.bframes
        args += ["-bf", str(bframes)]
        if bframes:
            args += ["-b_ref_mode", video.b_ref_mode]
    elif video.bframes is not None:
        args += ["-bf", str(video.bframes)]
    if video.gop_size:
        args += ["-g", str(video.gop_size)]
    if video.level:
        args += ["-level", video.level]
    return args


def _software_video_args(video: VideoSpec) -> list[str]:
    args: list[str] = []
    if video.codec == "av1":
        args += ["-preset", SVT_PRESET[video.preset]]
    else:
        args += ["-preset", X26X_PRESET[video.preset]]
    args += _rate_control_args(video, Pipeline.SOFTWARE)
    if video.profile and video.codec != "av1":
        args += ["-profile:v", video.profile]
    if video.bframes is not None and video.codec != "av1":
        args += ["-bf", str(video.bframes)]
    if video.gop_size:
        args += ["-g", str(video.gop_size)]
    if video.level and video.codec != "av1":
        args += ["-level", video.level]
    if video.codec == "hevc":
        args += ["-x265-params", "log-level=error"]
    return args


def _audio_args(render: RenderSpec, media: MediaInfo, warnings: list[str]) -> list[str]:
    audio = render.audio
    if audio.codec == "none":
        return ["-an"]
    if media.audio is None:
        warnings.append("Source has no audio stream; output will be silent.")
        return ["-an"]
    if audio.codec == "copy":
        allowed = CONTAINER_AUDIO_COPY_OK[render.container]
        if allowed is not None and media.audio.codec not in allowed:
            # Do not fail late inside FFmpeg with an obscure muxer error.
            from .errors import InputValidationError

            raise InputValidationError(
                f"audio.codec='copy' but the source audio codec {media.audio.codec!r} cannot be stored "
                f"in a .{render.container} container.",
                hint="Use audio.codec='aac' (re-encode) or container='mkv'.",
            )
        return ["-map", "0:a:0", "-c:a", "copy"]
    args = ["-map", "0:a:0", "-c:a", "aac", "-b:a", str(audio.bitrate_bps)]
    if audio.channels:
        args += ["-ac", str(audio.channels)]
    if audio.sample_rate:
        args += ["-ar", str(audio.sample_rate)]
    return args


def build_command(
    settings: Settings,
    render: RenderSpec,
    media: MediaInfo,
    pipeline: Pipeline,
    input_path: str,
    output_path: str,
) -> BuiltCommand:
    warnings: list[str] = []
    video = render.video
    geometry = compute_geometry(media, video, render.aspect)
    pix_fmt = video.effective_pixel_format

    args: list[str] = [settings.ffmpeg_bin, "-hide_banner", "-nostdin", "-y",
                       "-loglevel", "info", "-nostats", "-progress", "pipe:1"]

    # ---- input options -------------------------------------------------
    if pipeline is Pipeline.GPU_DECODE_NVENC:
        args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "16"]
    if render.trim and render.trim.start_seconds > 0:
        args += ["-ss", format_seconds(render.trim.start_seconds)]
    args += ["-i", input_path]

    # ---- stream selection / duration ----------------------------------
    args += ["-map", "0:v:0"]
    if render.trim and render.trim.end_seconds is not None:
        args += ["-t", format_seconds(render.trim.duration_seconds or 0)]

    # ---- video filters -------------------------------------------------
    filters: list[str] = []
    if pipeline is Pipeline.GPU_DECODE_NVENC:
        w, h = geometry.out_w, geometry.out_h
        filters.append(f"scale_cuda=w={w}:h={h}:format={CUDA_FORMAT[pix_fmt]}:interp_algo=lanczos")
    else:
        if geometry.crop:
            cw, ch, cx, cy = geometry.crop
            filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
        if geometry.pad:
            pw, ph, px, py = geometry.pad
            filters.append(f"pad={pw}:{ph}:{px}:{py}:color={render.aspect.pad_color if render.aspect else 'black'}")
        if geometry.scale:
            sw, sh = geometry.scale
            filters.append(f"scale={sw}:{sh}:flags=lanczos")
        if video.fps:
            filters.append(f"fps={video.fps:g}")
        filters.append(f"format={pix_fmt}")
    video_filter = ",".join(filters)
    args += ["-vf", video_filter]

    # ---- video encoder -------------------------------------------------
    if pipeline.uses_nvenc:
        encoder = NVENC_ENCODER[video.codec]
        args += ["-c:v", encoder] + _nvenc_video_args(video)
    else:
        encoder = SOFTWARE_ENCODER[video.codec]
        args += ["-c:v", encoder] + _software_video_args(video)
    if video.extra_args:
        args += list(video.extra_args)

    # ---- audio ---------------------------------------------------------
    args += _audio_args(render, media, warnings)

    # ---- container -----------------------------------------------------
    args += ["-map_metadata", "0" if render.options.keep_metadata else "-1", "-map_chapters", "-1"]
    if render.container in ("mp4", "mov"):
        if render.options.faststart:
            args += ["-movflags", "+faststart"]
        if video.codec == "hevc":
            args += ["-tag:v", "hvc1"]  # Apple/QuickTime compatibility
    args += ["-f", MUXER[render.container], output_path]

    return BuiltCommand(pipeline=pipeline, encoder=encoder, args=args,
                        video_filter=video_filter, geometry=geometry, warnings=warnings)
