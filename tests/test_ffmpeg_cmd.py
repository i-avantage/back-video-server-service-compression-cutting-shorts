import pytest

from video_compressor.config import Settings
from video_compressor.errors import InputValidationError
from video_compressor.ffmpeg_cmd import Pipeline, RenderSpec, build_command, compute_geometry, gpu_decode_eligible
from video_compressor.probe import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_compressor.schema import AspectSpec, AudioSpec, OptionsSpec, OutputSpec, TrimSpec, VideoSpec


def media(w=1920, h=1080, codec="h264", pix_fmt="yuv420p", rotation=0, audio="aac"):
    return MediaInfo(
        path="/in.mp4", format_name="mov,mp4", duration=60.0, size_bytes=100_000_000, bit_rate=13_000_000,
        video=VideoStreamInfo(codec=codec, width=w, height=h, pix_fmt=pix_fmt, fps=30.0, rotation=rotation),
        audio=AudioStreamInfo(codec=audio, channels=2, sample_rate=48000) if audio else None,
    )


def render(**kw):
    defaults = dict(name="out", filename="out.mp4", container="mp4", video=VideoSpec(), audio=AudioSpec(),
                    options=OptionsSpec(), output=OutputSpec(type="local", path="/o.mp4"))
    defaults.update(kw)
    return RenderSpec(**defaults)


SETTINGS = Settings(ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe")


def test_geometry_crop_9_16_center():
    geo = compute_geometry(media(), VideoSpec(), AspectSpec(ratio="9:16"))
    assert geo.crop == (608, 1080, 656, 0) and geo.scale is None and (geo.out_w, geo.out_h) == (608, 1080)


def test_geometry_crop_then_scale_to_1080x1920():
    geo = compute_geometry(media(), VideoSpec(width=1080, height=1920), AspectSpec(ratio="9:16"))
    assert geo.crop == (608, 1080, 656, 0) and geo.scale == (1080, 1920)


def test_geometry_pad_16_9_from_vertical():
    geo = compute_geometry(media(1080, 1920), VideoSpec(), AspectSpec(ratio="16:9", mode="pad"))
    assert geo.pad is not None and geo.pad[1] == 1920 and geo.pad[0] >= 3412 and geo.pad[0] % 2 == 0


def test_geometry_fit_never_upscales():
    geo = compute_geometry(media(1280, 720), VideoSpec(max_height=1080), None)
    assert geo.scale is None and (geo.out_w, geo.out_h) == (1280, 720)
    geo = compute_geometry(media(3840, 2160), VideoSpec(max_width=1920, max_height=1080), None)
    assert geo.scale == (1920, 1080)


def test_geometry_width_only_keeps_aspect_and_even():
    geo = compute_geometry(media(1920, 1080), VideoSpec(width=1000), None)
    assert geo.scale == (1000, 562)


def test_geometry_rotation_uses_display_dims():
    geo = compute_geometry(media(1920, 1080, rotation=90), VideoSpec(max_height=720), None)
    assert geo.scale == (404, 720)


def test_gpu_pipeline_command():
    cmd = build_command(SETTINGS, render(video=VideoSpec(max_height=720)), media(), Pipeline.GPU_DECODE_NVENC, "/in.mp4", "/o.mp4")
    args = cmd.args
    assert args[:1] == ["ffmpeg"] and "-hwaccel" in args and args[args.index("-hwaccel") + 1] == "cuda"
    assert cmd.video_filter.startswith("scale_cuda=w=1280:h=720:format=nv12")
    assert args[args.index("-c:v") + 1] == "h264_nvenc"
    assert "-rc" in args and args[args.index("-cq") + 1] == "23" and args[args.index("-b:v") + 1] == "0"
    assert "-multipass" in args and "-b_ref_mode" in args and "-movflags" in args
    assert args[-1] == "/o.mp4" and args[args.index("-f") + 1] == "mp4"
    assert "-progress" in args and "-nostdin" in args


def test_cpu_pipeline_with_crop_trim_and_hevc_tag():
    r = render(container="mp4", video=VideoSpec(codec="hevc", fps=24), aspect=AspectSpec(ratio="1:1"),
               trim=TrimSpec(start=5, end=15))
    cmd = build_command(SETTINGS, r, media(), Pipeline.CPU_DECODE_NVENC, "/in.mp4", "/o.mp4")
    args = cmd.args
    assert "-hwaccel" not in args
    assert args[args.index("-ss") + 1] == "00:00:05.000" and args.index("-ss") < args.index("-i")
    assert args[args.index("-t") + 1] == "00:00:10.000" and args.index("-t") > args.index("-i")
    assert cmd.video_filter == "crop=1080:1080:420:0,fps=24,format=yuv420p"
    assert args[args.index("-c:v") + 1] == "hevc_nvenc" and args[args.index("-tag:v") + 1] == "hvc1"


def test_software_pipeline_maps_presets_and_crf():
    cmd = build_command(SETTINGS, render(video=VideoSpec(preset="p7", cq=20)), media(), Pipeline.SOFTWARE, "/in.mp4", "/o.mp4")
    args = cmd.args
    assert args[args.index("-c:v") + 1] == "libx264" and args[args.index("-preset") + 1] == "slower"
    assert args[args.index("-crf") + 1] == "20" and "-rc" not in args


def test_vbr_and_cbr_args():
    v = VideoSpec(rate_control="vbr", bitrate="4M")
    args = build_command(SETTINGS, render(video=v), media(), Pipeline.CPU_DECODE_NVENC, "/i", "/o").args
    assert args[args.index("-b:v") + 1] == "4000000" and args[args.index("-maxrate") + 1] == "6000000"
    v = VideoSpec(rate_control="cbr", bitrate="4M")
    args = build_command(SETTINGS, render(video=v), media(), Pipeline.CPU_DECODE_NVENC, "/i", "/o").args
    assert args[args.index("-rc") + 1] == "cbr"


def test_audio_variants():
    args = build_command(SETTINGS, render(audio=AudioSpec(codec="none")), media(), Pipeline.SOFTWARE, "/i", "/o").args
    assert "-an" in args and "0:a:0" not in args
    args = build_command(SETTINGS, render(audio=AudioSpec(codec="copy")), media(), Pipeline.SOFTWARE, "/i", "/o").args
    assert args[args.index("-c:a") + 1] == "copy"
    cmd = build_command(SETTINGS, render(), media(audio=None), Pipeline.SOFTWARE, "/i", "/o")
    assert "-an" in cmd.args and cmd.warnings
    with pytest.raises(InputValidationError, match="cannot be stored"):
        build_command(SETTINGS, render(audio=AudioSpec(codec="copy")), media(audio="pcm_s16le"), Pipeline.SOFTWARE, "/i", "/o")


def test_gpu_decode_eligibility():
    r = render()
    assert gpu_decode_eligible(media(), r, compute_geometry(media(), r.video, None))[0]
    assert not gpu_decode_eligible(media(rotation=90), r, compute_geometry(media(rotation=90), r.video, None))[0]
    assert not gpu_decode_eligible(media(codec="prores", pix_fmt="yuv422p10le"), r, compute_geometry(media(), r.video, None))[0]
    r2 = render(aspect=AspectSpec(ratio="9:16"))
    assert not gpu_decode_eligible(media(), r2, compute_geometry(media(), r2.video, r2.aspect))[0]
    r3 = render(video=VideoSpec(fps=25))
    assert not gpu_decode_eligible(media(), r3, compute_geometry(media(), r3.video, None))[0]
