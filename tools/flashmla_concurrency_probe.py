# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send staged, non-streaming requests to an already running model service.

This client records HTTP evidence; it does not establish FlashMLA routing,
operator accuracy, device concurrency correctness, or performance acceptance.
"""

# Direct execution adds tools/ to sys.path; its bisect/ directory shadows the
# standard library needed by urllib. Isolate this standalone client first.
# ruff: noqa: E402
import os
import sys

if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import concurrent.futures
import datetime
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep authorization and requests on the explicitly selected endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Explicit API root, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate-sha", required=True, help="SHA reported by the release-machine checkout")
    parser.add_argument("--api", choices=("completions", "chat/completions"), default="completions")
    parser.add_argument("--concurrency", default="1,4,16,32")
    parser.add_argument("--requests-per-stage", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0, help="Socket timeout in seconds, not a run deadline")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Optional bearer token environment variable")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 prompt; contents are not written to reports")
    prompt.add_argument("--prompts-json", type=Path, help="UTF-8 JSON list of prompts, used round-robin")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="A new directory; never overwrite earlier evidence"
    )
    args = parser.parse_args()
    try:
        args.concurrency = [int(value) for value in args.concurrency.split(",")]
    except ValueError:
        parser.error("concurrency must be comma-separated positive integers")
    if not args.concurrency or any(value <= 0 for value in args.concurrency):
        parser.error("concurrency must be comma-separated positive integers")
    if args.requests_per_stage <= 0 or args.max_tokens <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("requests-per-stage, max-tokens, and timeout must be positive and finite")
    if any(value > args.requests_per_stage for value in args.concurrency):
        parser.error("requests-per-stage must be at least the largest requested concurrency")
    parsed = urllib.parse.urlsplit(args.base_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        parser.error("base-url must be HTTP(S) without user info, query, or fragment")
    if not 7 <= len(args.candidate_sha) <= 40 or any(c not in "0123456789abcdefABCDEF" for c in args.candidate_sha):
        parser.error("candidate-sha must be a 7-40 character hexadecimal Git revision")
    try:
        if args.prompts_json:
            args.prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))
        else:
            args.prompts = [args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt]
    except (OSError, UnicodeError, ValueError):
        parser.error("cannot read prompt file as UTF-8 text or JSON")
    if (
        not isinstance(args.prompts, list)
        or not args.prompts
        or any(not isinstance(value, str) or not value.strip() for value in args.prompts)
    ):
        parser.error("prompts must be a nonempty list of nonempty strings")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")
    args.base_url = args.base_url.rstrip("/")
    return args


def check_response(body, api):
    """Check one non-streaming choice without persisting generated text."""
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or len(body["choices"]) != 1:
        raise ValueError("invalid_choices")
    choice = body["choices"][0]
    if not isinstance(choice, dict):
        raise ValueError("invalid_choice")
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length", "content_filter", "tool_calls", "function_call"):
        raise ValueError("missing_or_unknown_finish_reason")
    tool_output = False
    if api == "completions":
        content = choice.get("text")
        if not isinstance(content, str):
            raise ValueError("missing_completion_text")
    else:
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("missing_chat_message")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("invalid_chat_content")
        tool_output = bool(message.get("tool_calls") or message.get("function_call"))
    nonempty = bool(content and content.strip())
    if finish in ("tool_calls", "function_call") and not tool_output:
        raise ValueError("missing_tool_output")
    usage = body.get("usage")
    safe_usage = {
        key: value
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(usage, dict)
        and isinstance(value := usage.get(key), int)
        and not isinstance(value, bool)
        and value >= 0
    }
    return {
        "finish_reason": finish,
        "nonempty_text": nonempty,
        "response_observation": "text" if nonempty else "tool_output" if tool_output else "empty_terminal_response",
        "usage": safe_usage,
    }


def send_request(args, run_id, stage, index, payload, token):
    request_id = f"flashmla-{run_id}-s{stage}-r{index}"
    record = {
        "type": "request",
        "run_id": run_id,
        "request_id": request_id,
        "stage": stage,
        "prompt_index": index % len(args.prompts),
        "concurrency": args.concurrency[stage],
        "started_utc": utc_now(),
        "http_status": None,
        "ok": False,
    }
    started = time.monotonic()
    headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{args.base_url}/{args.api}", data=payload, headers=headers, method="POST")
    try:
        # No retries or redirects: each evidence row corresponds to one attempt.
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=args.timeout) as response:
            record["http_status"] = response.status
            raw = response.read()
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            record["error"] = "invalid_json"
        else:
            try:
                record.update(check_response(body, args.api))
                record["ok"] = True
            except ValueError as error:
                record["error"] = str(error)  # Only the fixed validation codes above.
    except urllib.error.HTTPError as error:
        record["http_status"] = error.code
        record["error"] = "http_error"
        error.close()  # Do not persist the response body (may echo the prompt).
    except Exception as error:
        # Exception text can contain request data, URLs, or proxy credentials.
        record["error"] = type(error).__name__
        if isinstance(error, urllib.error.URLError):
            record["error_cause"] = type(error.reason).__name__
    record["ended_utc"] = utc_now()
    record["latency_s"] = time.monotonic() - started
    return record


def stage_summary(records, stage, concurrency, elapsed):
    latencies = sorted(record["latency_s"] for record in records)
    last = max(records, key=lambda record: record["ended_utc"])
    return {
        "type": "stage_summary",
        "stage": stage,
        "concurrency": concurrency,
        "requests": len(records),
        "successful_responses": sum(record["ok"] for record in records),
        "failed_responses": sum(not record["ok"] for record in records),
        "empty_terminal_responses": sum(
            record.get("response_observation") == "empty_terminal_response" for record in records
        ),
        "elapsed_s": elapsed,
        "all_attempt_latency_s": {
            "min": latencies[0],
            "median": (latencies[(len(latencies) - 1) // 2] + latencies[len(latencies) // 2]) / 2,
            "p95_nearest_rank": latencies[math.ceil(len(latencies) * 0.95) - 1],
            "max": latencies[-1],
        },
        "last_completed_request_id": last["request_id"],
        "last_completed_utc": last["ended_utc"],
    }


def main():
    args = parse_args()
    run_id = uuid.uuid4().hex[:12]
    encoded = []
    for prompt in args.prompts:
        payload = {"model": args.model, "max_tokens": args.max_tokens, "stream": False, "temperature": 0, "n": 1}
        if args.api == "completions":
            payload["prompt"] = prompt
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        encoded.append(json.dumps(payload).encode("utf-8"))
    token = os.environ.get(args.api_key_env)
    report = {
        "type": "run",
        "run_id": run_id,
        "started_utc": utc_now(),
        "candidate_sha_reported": args.candidate_sha,
        "candidate_sha_verified_by_client": False,
        "base_url": args.base_url,
        "model": args.model,
        "api": args.api,
        "concurrency_stages": args.concurrency,
        "requests_per_stage": args.requests_per_stage,
        "socket_timeout_s": args.timeout,
        "max_tokens": args.max_tokens,
        "prompt_count": len(args.prompts),
        "stages": [],
        "scope": "HTTP response structure only; no operator, graph, numerical, or performance acceptance",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "requests.jsonl").open("x", encoding="utf-8") as output:

        def write_record(record):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

        write_record({key: value for key, value in report.items() if key != "stages"})
        for stage, concurrency in enumerate(args.concurrency):
            started = time.monotonic()
            records = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(send_request, args, run_id, stage, index, encoded[index % len(encoded)], token)
                    for index in range(args.requests_per_stage)
                ]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    records.append(record)
                    write_record(record)
            summary = stage_summary(records, stage, concurrency, time.monotonic() - started)
            report["stages"].append(summary)
            write_record(summary)
            (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary), flush=True)
            if summary["failed_responses"]:
                report["stopped_after_failed_stage"] = True
                break
    report["ended_utc"] = utc_now()
    report["last_completed_request_id"] = report["stages"][-1]["last_completed_request_id"]
    report["last_completed_utc"] = report["stages"][-1]["last_completed_utc"]
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if any(stage["failed_responses"] for stage in report["stages"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
