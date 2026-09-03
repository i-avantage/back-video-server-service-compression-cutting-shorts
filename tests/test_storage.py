import http.server
import threading

import pytest

from video_compressor.config import Settings
from video_compressor.errors import (
    DownloadError,
    InputValidationError,
    OutputExistsError,
    StorageConfigError,
    UploadError,
)
from video_compressor.schema import OutputSpec
from video_compressor.storage import copy_local, download_http, inline_base64, parse_source, resolve_output


def test_parse_source():
    assert parse_source("https://h/p/video.mp4?sig=1").kind == "http"
    assert parse_source("https://h/p/video.mp4?sig=1").filename == "video.mp4"
    s = parse_source("s3://bucket/dir/file.mov")
    assert (s.kind, s.bucket, s.key, s.filename) == ("s3", "bucket", "dir/file.mov", "file.mov")
    assert parse_source("/data/in.mkv").kind == "local"
    with pytest.raises(InputValidationError):
        parse_source("s3://bucket-only")


def test_resolve_output_defaults(monkeypatch):
    settings = Settings.from_env({"S3_BUCKET": "out-bucket", "S3_OUTPUT_PREFIX": "enc/"})
    out = resolve_output(None, settings, job_id="job1", filename="a.mp4", container="mp4")
    assert out.type == "s3" and out.bucket == "out-bucket" and out.key == "enc/job1/a.mp4"
    out = resolve_output(OutputSpec(type="s3", key="x/y.mp4"), settings, job_id="j", filename="a.mp4", container="mp4")
    assert out.bucket == "out-bucket" and out.key == "x/y.mp4"
    with pytest.raises(StorageConfigError):
        resolve_output(None, Settings.from_env({}), job_id="j", filename="a.mp4", container="mp4")
    with pytest.raises(StorageConfigError):
        resolve_output(OutputSpec(type="s3"), Settings.from_env({}), job_id="j", filename="a.mp4", container="mp4")


def test_copy_local_and_overwrite(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"data")
    dest = tmp_path / "out" / "file.mp4"
    result = copy_local(str(src), str(dest), overwrite=False)
    assert result["path"] == str(dest) and dest.read_bytes() == b"data"
    with pytest.raises(OutputExistsError):
        copy_local(str(src), str(dest), overwrite=False)
    copy_local(str(src), str(dest), overwrite=True)


def test_inline_base64_cap(tmp_path):
    f = tmp_path / "f.mp4"
    f.write_bytes(b"x" * 100)
    assert inline_base64(str(f), Settings(max_inline_output_bytes=1000))["size_bytes"] == 100
    with pytest.raises(UploadError) as exc:
        inline_base64(str(f), Settings(max_inline_output_bytes=10))
    assert exc.value.code == "OUTPUT_TOO_LARGE_FOR_INLINE"


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = b"\x00\x00\x00\x1cftypisom" + b"v" * 5000

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/ok.mp4":
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
        elif path == "/expired.mp4":
            body = b"<Error><Code>AccessDenied</Code></Error>"
            self.send_response(403)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = b"<html>login page</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def http_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_download_http_ok(http_server, tmp_path):
    settings = Settings(download_attempts=1)
    dest = tmp_path / "in.mp4"
    n = download_http(f"{http_server}/ok.mp4", str(dest), settings)
    assert n == len(_Handler.payload) and dest.read_bytes() == _Handler.payload


def test_download_http_403_is_explicit(http_server, tmp_path):
    with pytest.raises(DownloadError) as exc:
        download_http(f"{http_server}/expired.mp4?sig=1", str(tmp_path / "x"), Settings(download_attempts=1))
    assert exc.value.details["status_code"] == 403 and "AccessDenied" in exc.value.details["body_snippet"]
    assert "<redacted>" in exc.value.message


def test_download_http_html_page_rejected(http_server, tmp_path):
    with pytest.raises(DownloadError, match="instead of a video"):
        download_http(f"{http_server}/login", str(tmp_path / "x"), Settings(download_attempts=1))


def test_download_http_size_limit(http_server, tmp_path):
    with pytest.raises(DownloadError) as exc:
        download_http(f"{http_server}/ok.mp4", str(tmp_path / "x"), Settings(download_attempts=1, max_input_bytes=100))
    assert exc.value.code == "INPUT_TOO_LARGE"
