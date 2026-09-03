"""GPU (NVENC) video compression / cutting service for RunPod Serverless.

The heavy lifting is done by the FFmpeg CLI (invoked via ``subprocess``);
this package is the orchestration layer: input validation, download,
FFmpeg command construction, execution with progress/timeouts, output
verification and delivery to object storage.
"""

__version__ = "1.0.0"
