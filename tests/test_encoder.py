import sys

import pytest

from video_compressor.capabilities import Capabilities
from video_compressor.config import Settings
from video_compressor.encoder import classify_failure, plan_pipelines, run_ffmpeg
from video_compressor.errors import EncodeError, GpuError
from video_compressor.ffmpeg_cmd import Pipeline, RenderSpec
from video_compressor.probe import MediaInfo, VideoStreamInfo
from video_compressor.schema import AudioSpec, OptionsSpec, OutputSpec, VideoSpec


@pytest.mark.parametrize("text,code", [
    ("[h264_nvenc @ 0x1] Cannot load libnvidia-encode.so.1", "GPU_LIBRARIES_NOT_VISIBLE"),
    ("Driver does not support the required nvenc API version. Required: 13.0 Found: 12.1", "NVENC_DRIVER_TOO_OLD"),
    ("[av1_nvenc @ 0x1] No capable devices found", "NVENC_NOT_SUPPORTED_ON_GPU"),
    ("OpenEncodeSessionEx failed: out of memory (10)", "NVENC_SESSION_LIMIT"),
    ("Impossible to convert between the formats supported by the filter", "HW_DECODE_FAILED"),
    ("[mov,mp4,m4a @ 0x1] moov atom not found", "INPUT_INVALID"),
    ("av_interleaved_write_frame(): No space left on device", "DISK_FULL"),
    ("Unknown encoder 'libx265'", "ENCODER_MISSING"),
    ("something odd happened", "ENCODE_FAILED"),
])
def test_classify(text, code):
    assert classify_failure(text, 1).code == code


def test_classify_sigkill():
    assert classify_failure("", -9).code == "PROCESS_KILLED"


def _media(rotation=0):
    return MediaInfo(path="/i", format_name="mp4", duration=10, size_bytes=1000, bit_rate=None,
                     video=VideoStreamInfo("h264", 1280, 720, "yuv420p", 30.0, rotation), audio=None)


def _render(**opts):
    return RenderSpec(name="n", filename="n.mp4", container="mp4", video=VideoSpec(), audio=AudioSpec(),
                      options=OptionsSpec(**opts), output=OutputSpec(type="local", path="/o.mp4"))


def test_plan_software_backend():
    plan, _ = plan_pipelines(Settings(encoder_backend="software"), Capabilities(encoders={"libx264"}), _render(), _media())
    assert plan == [Pipeline.SOFTWARE]


def test_plan_nvenc_full_chain_and_fallback():
    caps = Capabilities(encoders={"h264_nvenc", "libx264"}, nvenc_available=True)
    plan, _ = plan_pipelines(Settings(encoder_backend="nvenc"), caps, _render(), _media())
    assert plan == [Pipeline.GPU_DECODE_NVENC, Pipeline.CPU_DECODE_NVENC]
    plan, _ = plan_pipelines(Settings(encoder_backend="nvenc"), caps, _render(fallback_to_software=True), _media())
    assert plan[-1] == Pipeline.SOFTWARE
    plan, notes = plan_pipelines(Settings(encoder_backend="nvenc"), caps, _render(), _media(rotation=90))
    assert plan == [Pipeline.CPU_DECODE_NVENC] and "rotation" in notes[0]
    plan, _ = plan_pipelines(Settings(encoder_backend="nvenc"), caps, _render(hw_decode="off"), _media())
    assert plan == [Pipeline.CPU_DECODE_NVENC]
    plan, _ = plan_pipelines(Settings(encoder_backend="nvenc"), caps, _render(hw_decode="on"), _media())
    assert plan == [Pipeline.GPU_DECODE_NVENC]
    with pytest.raises(EncodeError) as exc:
        plan_pipelines(Settings(encoder_backend="nvenc"), caps, _render(hw_decode="on"), _media(rotation=90))
    assert exc.value.code == "HW_DECODE_NOT_POSSIBLE"


def test_plan_strict_nvenc_missing_raises_classified_gpu_error():
    caps = Capabilities(encoders={"h264_nvenc", "libx264"}, nvenc_available=False,
                        nvenc_error="Cannot load libnvidia-encode.so.1", nvenc_error_log="[x] Cannot load libnvidia-encode.so.1")
    with pytest.raises(GpuError) as exc:
        plan_pipelines(Settings(encoder_backend="nvenc"), caps, _render(), _media())
    assert exc.value.code == "GPU_LIBRARIES_NOT_VISIBLE" and exc.value.refresh_worker
    plan, notes = plan_pipelines(Settings(encoder_backend="auto"), caps, _render(), _media())
    assert plan == [Pipeline.SOFTWARE] and "NVENC unavailable" in notes[0]


FAKE_FFMPEG = """
import sys, time
sys.stdout.write("frame=10\\nfps=25.0\\nout_time_us=1000000\\nspeed=1.0x\\nprogress=continue\\n"); sys.stdout.flush()
sys.stderr.write("[warn] something\\n")
if "--sleep" in sys.argv: time.sleep(30)
sys.stdout.write("frame=20\\nout_time_us=2000000\\nprogress=end\\n"); sys.stdout.flush()
sys.stderr.write("fatal: boom\\n")
sys.exit(3 if "--fail" in sys.argv else 0)
"""


def test_run_ffmpeg_parses_progress_and_stderr():
    seen = []
    run = run_ffmpeg([sys.executable, "-c", FAKE_FFMPEG, "--fail"], expected_duration=4.0,
                     progress_cb=seen.append, progress_interval=0)
    assert run.returncode == 3 and not run.ok
    assert run.last_progress["out_time_seconds"] == 2.0 and run.last_progress["percent"] == 50.0
    assert seen and seen[0]["frame"] == 10
    assert "fatal: boom" in run.stderr_tail


def test_run_ffmpeg_timeout_kills_process():
    run = run_ffmpeg([sys.executable, "-c", FAKE_FFMPEG, "--sleep"], timeout_seconds=1)
    assert run.timed_out and not run.ok and run.elapsed_seconds < 10
