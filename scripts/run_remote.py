#!/usr/bin/env python3
"""Submit a job to a deployed RunPod Serverless endpoint and wait for the result.

    export RUNPOD_API_KEY=...          # RunPod -> Settings -> API Keys
    export RUNPOD_ENDPOINT_ID=...      # RunPod -> Serverless -> your endpoint
    python scripts/run_remote.py examples/compress_to_s3.json
    python scripts/run_remote.py examples/cut_segment.json --timeout-minutes 60

Uses the plain HTTPS API (no SDK needed): /run to submit, /status/<id> to poll.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.ai/v2"


def _request(method: str, url: str, api_key: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code} from RunPod: {exc.read().decode()[:1000]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("payload", help="JSON file with {'input': {...}} (see examples/)")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--timeout-minutes", type=float, default=None,
                        help="Sets policy.executionTimeout for this job (default: endpoint setting)")
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if not args.endpoint_id or not args.api_key:
        sys.exit("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY (or pass --endpoint-id/--api-key).")

    with open(args.payload, encoding="utf-8") as fh:
        payload = json.load(fh)
    if "input" not in payload:
        sys.exit("The payload file must contain a top-level 'input' object.")
    if args.timeout_minutes:
        payload.setdefault("policy", {})["executionTimeout"] = int(args.timeout_minutes * 60_000)

    base = f"{API}/{args.endpoint_id}"
    submitted = _request("POST", f"{base}/run", args.api_key, payload)
    job_id = submitted.get("id")
    if not job_id:
        sys.exit(f"Unexpected response from /run: {submitted}")
    print(f"Submitted job {job_id} (status {submitted.get('status')})")

    started = time.time()
    last_progress = None
    while True:
        status = _request("GET", f"{base}/status/{job_id}", args.api_key)
        state = status.get("status")
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        progress = status.get("output")
        if progress and progress != last_progress:
            last_progress = progress
            print(f"  [{int(time.time() - started):4d}s] {state}: {json.dumps(progress)}")
        elif int(time.time() - started) % 30 == 0:
            print(f"  [{int(time.time() - started):4d}s] {state}")
        time.sleep(args.poll_seconds)

    print(f"\nFinal status: {state} after {int(time.time() - started)}s")
    print(json.dumps(status, indent=2))
    if state != "COMPLETED":
        return 1
    output = status.get("output") or {}
    if output.get("status") != "success":
        return 1
    for out in output.get("outputs", []):
        dest = out.get("destination", {})
        print(f"\nOK {out['name']}: {dest.get('url') or dest.get('uri') or dest.get('path')} "
              f"({out['size_bytes']} bytes, -{out['size_reduction_percent']}%, {out['pipeline']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
