import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import fitz


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.getenv("CHAT_TEST_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_CONCURRENCY_STEPS = [5, 10, 15, 20, 25]
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_LOG_DIR = ROOT_DIR / "scripts" / "test_logs"
DEFAULT_STAGE_WINDOW_SECONDS = 30

PDF_BATCH_CONFIGS = [
    {
        "pdf_path": ROOT_DIR / "8a861151-5d27-4daf-bb1b-6cf9deeff22c.pdf",
        "question_bank_path": ROOT_DIR
        / "scripts"
        / "question_banks"
        / "8a861151-5d27-4daf-bb1b-6cf9deeff22c_rewritten_questions.md",
        "batch_id": "8a861151-5d27-4daf-bb1b-6cf9deeff22c",
        "username": "nithish-learner",
    },
    {
        "pdf_path": ROOT_DIR / "b33f8473-db92-494d-9216-0cde7e1854e5.pdf",
        "question_bank_path": ROOT_DIR
        / "scripts"
        / "question_banks"
        / "b33f8473-db92-494d-9216-0cde7e1854e5_rewritten_questions.md",
        "batch_id": "b33f8473-db92-494d-9216-0cde7e1854e5",
        "username": "sathish-learner",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run concurrent /chat load tests using the rewritten question banks for the provided batches."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--concurrency-steps",
        default="5,10,15,20,25",
        help="Comma-separated concurrency stages to run sequentially",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds")
    parser.add_argument(
        "--stage-window-seconds",
        type=int,
        default=DEFAULT_STAGE_WINDOW_SECONDS,
        help="Spread each stage's request submissions across this many seconds",
    )
    parser.add_argument("--complexity", default="Undergrad", help="Complexity to send in the chat payload")
    parser.add_argument("--tutor-mode", action="store_true", help="Enable tutor_mode=true in the chat payload")
    parser.add_argument("--limit-per-batch", type=int, default=0, help="Optional cap on questions per batch (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Only extract and list questions without sending requests")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory where detailed log files will be written")
    return parser.parse_args()


def sanitize_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_concurrency_steps(raw_value: str) -> List[int]:
    steps: List[int] = []
    for part in str(raw_value or "").split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Concurrency stages must be positive integers.")
        steps.append(value)

    if not steps:
        raise ValueError("At least one concurrency stage must be provided.")

    return steps


def extract_questions_from_pdf(pdf_path: Path) -> List[str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Question PDF not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        lines: List[str] = []
        for page in doc:
            for raw_line in page.get_text().splitlines():
                line = sanitize_text(raw_line)
                if line:
                    lines.append(line)

    questions: List[str] = []
    current_question = ""

    for line in lines:
        match = re.match(r"^(\d{1,3})\.\s*(.*)$", line)
        if match:
            if current_question:
                questions.append(sanitize_text(current_question))
            current_question = sanitize_text(match.group(2))
            continue

        if current_question:
            current_question = f"{current_question} {line}"

    if current_question:
        questions.append(sanitize_text(current_question))

    return [question for question in questions if question]


def extract_questions_from_markdown(question_bank_path: Path) -> List[str]:
    if not question_bank_path.exists():
        raise FileNotFoundError(f"Question bank not found: {question_bank_path}")

    lines = [sanitize_text(line) for line in question_bank_path.read_text(encoding="utf-8").splitlines()]
    questions: List[str] = []
    current_question = ""

    for line in lines:
        match = re.match(r"^(\d{1,3})\.\s*(.*)$", line)
        if match:
            if current_question:
                questions.append(sanitize_text(current_question))
            current_question = sanitize_text(match.group(2))
            continue

        if current_question and line:
            current_question = f"{current_question} {line}"

    if current_question:
        questions.append(sanitize_text(current_question))

    return [question for question in questions if question]


def load_questions_for_config(config: Dict) -> List[str]:
    question_bank_path = config.get("question_bank_path")
    if question_bank_path:
        return extract_questions_from_markdown(Path(question_bank_path))
    return extract_questions_from_pdf(Path(config["pdf_path"]))


def build_jobs(limit_per_batch: int) -> List[Dict]:
    per_batch_jobs: List[List[Dict]] = []

    for config in PDF_BATCH_CONFIGS:
        questions = load_questions_for_config(config)
        if limit_per_batch > 0:
            questions = questions[:limit_per_batch]

        batch_jobs = []
        for index, question in enumerate(questions, start=1):
            batch_jobs.append(
                {
                    "request_id": f"{config['batch_id']}-q{index:03d}",
                    "question_index": index,
                    "question": question,
                    "question_source_path": str(config["question_bank_path"]),
                    "question_source_name": Path(config["question_bank_path"]).name,
                    "pdf_path": str(config["pdf_path"]),
                    "pdf_name": config["pdf_path"].name,
                    "batch_id": config["batch_id"],
                    "username": config["username"],
                }
            )
        per_batch_jobs.append(batch_jobs)

    interleaved: List[Dict] = []
    max_len = max((len(batch_jobs) for batch_jobs in per_batch_jobs), default=0)
    for offset in range(max_len):
        for batch_jobs in per_batch_jobs:
            if offset < len(batch_jobs):
                interleaved.append(batch_jobs[offset])

    return interleaved


def send_chat_request(
    *,
    base_url: str,
    timeout_seconds: int,
    complexity: str,
    tutor_mode: bool,
    job: Dict,
) -> Dict:
    started_at = datetime.now(timezone.utc).isoformat()
    perf_start = time.perf_counter()

    payload = {
        "username": job["username"],
        "active_batch_id": job["batch_id"],
        "message": job["question"],
        "complexity": complexity,
        "tutor_mode": tutor_mode,
    }

    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    result = {
        **job,
        "started_at": started_at,
        "status": "failed",
        "http_status": None,
        "client_total_ms": None,
        "server_total_ms": None,
        "server_debug_timings": None,
        "reply_length": 0,
        "reply_preview": "",
        "error": None,
    }

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            data = json.loads(response_body)
            client_total_ms = round((time.perf_counter() - perf_start) * 1000, 2)

            reply = data.get("reply")
            reply_text = "" if reply is None else str(reply)

            result.update(
                {
                    "status": "success",
                    "http_status": response.status,
                    "client_total_ms": client_total_ms,
                    "server_total_ms": data.get("debug_timings", {}).get("total_ms"),
                    "server_debug_timings": data.get("debug_timings"),
                    "reply_length": len(reply_text),
                    "reply_preview": reply_text[:240],
                }
            )
            return result
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        error_detail = body
        try:
            data = json.loads(body)
            error_detail = data.get("detail") or body
        except json.JSONDecodeError:
            pass

        result.update(
            {
                "http_status": exc.code,
                "client_total_ms": round((time.perf_counter() - perf_start) * 1000, 2),
                "error": sanitize_text(error_detail),
            }
        )
        return result
    except Exception as exc:
        result.update(
            {
                "client_total_ms": round((time.perf_counter() - perf_start) * 1000, 2),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result


def ensure_log_dir(path_str: str) -> Path:
    path = Path(path_str).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary(summary_path: Path, summary: Dict):
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_stage_plan(jobs: List[Dict], concurrency_steps: List[int]) -> List[List[Dict]]:
    stage_jobs: List[List[Dict]] = []
    cursor = 0

    for stage_concurrency in concurrency_steps:
        next_cursor = cursor + stage_concurrency
        slice_jobs = jobs[cursor:next_cursor]
        if len(slice_jobs) < stage_concurrency:
            raise ValueError(
                f"Not enough questions to satisfy stage size {stage_concurrency}. "
                f"Needed {stage_concurrency}, found {len(slice_jobs)}."
            )
        stage_jobs.append(slice_jobs)
        cursor = next_cursor

    return stage_jobs


def run_stage(
    *,
    stage_concurrency: int,
    stage_index: int,
    stage_count: int,
    jobs: List[Dict],
    args: argparse.Namespace,
    log_dir: Path,
    run_stamp: str,
    manifest: Dict,
) -> Dict:
    detailed_log_path = log_dir / (
        f"chat_pdf_load_test_{run_stamp}_stage{stage_index:02d}_concurrency_{stage_concurrency}.jsonl"
    )
    summary_log_path = log_dir / (
        f"chat_pdf_load_test_{run_stamp}_stage{stage_index:02d}_concurrency_{stage_concurrency}_summary.json"
    )
    stage_window_seconds = max(0, int(args.stage_window_seconds))
    launch_interval_seconds = (
        float(stage_window_seconds) / float(max(1, len(jobs) - 1))
        if len(jobs) > 1 and stage_window_seconds > 0
        else 0.0
    )

    print(
        f"\n=== Stage {stage_index}/{stage_count} | concurrency={stage_concurrency} ===\n"
        f"Detailed log file: {detailed_log_path}\n"
        f"Summary log file:  {summary_log_path}\n"
        f"Stage requests:     {len(jobs)}\n"
        f"Submission window:  {stage_window_seconds}s\n"
        f"Launch interval:    {launch_interval_seconds:.2f}s"
    )

    run_started_at = datetime.now(timezone.utc).isoformat()
    results: List[Dict] = []

    with detailed_log_path.open("w", encoding="utf-8") as log_file:
        with ThreadPoolExecutor(max_workers=stage_concurrency) as executor:
            future_map = {}
            launch_start = time.perf_counter()

            for job_index, job in enumerate(jobs):
                target_offset = launch_interval_seconds * job_index
                elapsed = time.perf_counter() - launch_start
                remaining = target_offset - elapsed
                if remaining > 0:
                    time.sleep(remaining)

                future = executor.submit(
                    send_chat_request,
                    base_url=args.base_url,
                    timeout_seconds=args.timeout,
                    complexity=args.complexity,
                    tutor_mode=args.tutor_mode,
                    job=job,
                )
                future_map[future] = job

            for future in as_completed(future_map):
                result = future.result()
                result["stage_concurrency"] = stage_concurrency
                result["stage_index"] = stage_index
                results.append(result)
                log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                log_file.flush()

                status = result["status"].upper()
                duration_ms = result.get("client_total_ms")
                print(
                    f"[{status}][C={stage_concurrency}] {result['request_id']} "
                    f"user={result['username']} batch={result['batch_id']} "
                    f"question={result['question_index']} client_total_ms={duration_ms} "
                    f"http_status={result['http_status']}"
                )

    success_results = [item for item in results if item["status"] == "success"]
    failed_results = [item for item in results if item["status"] != "success"]
    client_durations = [item["client_total_ms"] for item in success_results if item["client_total_ms"] is not None]
    server_durations = [item["server_total_ms"] for item in success_results if item["server_total_ms"] is not None]

    summary = {
        "mode": "execute",
        "run_started_at": run_started_at,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "stage_index": stage_index,
        "stage_count": stage_count,
        "concurrency": stage_concurrency,
        "stage_window_seconds": stage_window_seconds,
        "launch_interval_seconds": round(launch_interval_seconds, 2),
        "base_url": args.base_url,
        "complexity": args.complexity,
        "tutor_mode": args.tutor_mode,
        "timeout_seconds": args.timeout,
        "total_jobs": len(jobs),
        "success_count": len(success_results),
        "failure_count": len(failed_results),
        "avg_client_total_ms": round(sum(client_durations) / len(client_durations), 2) if client_durations else None,
        "max_client_total_ms": round(max(client_durations), 2) if client_durations else None,
        "avg_server_total_ms": round(sum(server_durations) / len(server_durations), 2) if server_durations else None,
        "max_server_total_ms": round(max(server_durations), 2) if server_durations else None,
        "detailed_log_file": str(detailed_log_path),
        "manifest": manifest,
        "failures": [
            {
                "request_id": item["request_id"],
                "batch_id": item["batch_id"],
                "username": item["username"],
                "question_index": item["question_index"],
                "http_status": item["http_status"],
                "error": item["error"],
            }
            for item in failed_results
        ],
    }

    write_summary(summary_log_path, summary)
    print(f"Stage {stage_index}/{stage_count} completed.")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    concurrency_steps = parse_concurrency_steps(args.concurrency_steps)
    jobs = build_jobs(args.limit_per_batch)
    stage_plan = build_stage_plan(jobs, concurrency_steps)
    log_dir = ensure_log_dir(args.log_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = {}
    for config in PDF_BATCH_CONFIGS:
        extracted = load_questions_for_config(config)
        if args.limit_per_batch > 0:
            extracted = extracted[:args.limit_per_batch]
        manifest[config["batch_id"]] = {
            "username": config["username"],
            "question_source_name": Path(config["question_bank_path"]).name,
            "pdf_name": config["pdf_path"].name,
            "question_count": len(extracted),
            "sample_question": extracted[0] if extracted else None,
        }

    print("Chat PDF Load Test Manifest")
    print(json.dumps(manifest, indent=2))
    print(f"Concurrency stages: {concurrency_steps}")
    print(f"Stage window seconds: {args.stage_window_seconds}")

    if args.dry_run:
        summary_log_path = log_dir / f"chat_pdf_load_test_{run_stamp}_dry_run_summary.json"
        summary = {
            "mode": "dry_run",
            "base_url": args.base_url,
            "concurrency_steps": concurrency_steps,
            "stage_window_seconds": args.stage_window_seconds,
            "complexity": args.complexity,
            "tutor_mode": args.tutor_mode,
            "total_jobs": len(jobs),
            "total_jobs_planned": sum(len(stage_jobs) for stage_jobs in stage_plan),
            "stage_request_counts": [len(stage_jobs) for stage_jobs in stage_plan],
            "manifest": manifest,
        }
        write_summary(summary_log_path, summary)
        print("Dry run completed. No requests were sent.")
        return

    all_stage_summaries = []
    for stage_index, (stage_concurrency, stage_jobs) in enumerate(zip(concurrency_steps, stage_plan), start=1):
        all_stage_summaries.append(
            run_stage(
                stage_concurrency=stage_concurrency,
                stage_index=stage_index,
                stage_count=len(concurrency_steps),
                jobs=stage_jobs,
                args=args,
                log_dir=log_dir,
                run_stamp=run_stamp,
                manifest=manifest,
            )
        )

    aggregate_summary_path = log_dir / f"chat_pdf_load_test_{run_stamp}_all_stages_summary.json"
    aggregate_summary = {
        "mode": "multi_stage_execute",
        "base_url": args.base_url,
        "concurrency_steps": concurrency_steps,
        "stage_window_seconds": args.stage_window_seconds,
        "complexity": args.complexity,
        "tutor_mode": args.tutor_mode,
        "timeout_seconds": args.timeout,
        "total_jobs_available": len(jobs),
        "total_jobs_planned": sum(len(stage_jobs) for stage_jobs in stage_plan),
        "stage_request_counts": [len(stage_jobs) for stage_jobs in stage_plan],
        "manifest": manifest,
        "stage_summaries": all_stage_summaries,
    }
    write_summary(aggregate_summary_path, aggregate_summary)
    print("\nAll stages completed.")
    print(f"Aggregate summary file: {aggregate_summary_path}")
    print(json.dumps(aggregate_summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
