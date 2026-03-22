import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.getenv("CHAT_TEST_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_CONCURRENCY_STEPS = [5, 10, 15, 20, 25]
DEFAULT_STAGE_WINDOW_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_JOB_TIMEOUT_SECONDS = 900
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_LOG_DIR = ROOT_DIR / "scripts" / "test_logs"
DEFAULT_COMPLEXITIES = ["5-Year-Old", "High School", "Undergrad", "PhD Expert"]
DEFAULT_TOOL_SEQUENCE = ["chat", "flashcards", "quiz", "podcast"]

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
        description=(
            "Run staged mixed load against chat and studio tools using rewritten question banks, "
            "with randomized complexity per request."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--concurrency-steps",
        default="5,10,15,20,25",
        help="Comma-separated stage sizes to run sequentially",
    )
    parser.add_argument(
        "--stage-window-seconds",
        type=int,
        default=DEFAULT_STAGE_WINDOW_SECONDS,
        help="Spread each stage's request submissions across this many seconds",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds")
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=DEFAULT_JOB_TIMEOUT_SECONDS,
        help="Maximum wait time for asynchronous jobs like podcast generation",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="Polling interval in seconds for background jobs",
    )
    parser.add_argument(
        "--complexities",
        default=",".join(DEFAULT_COMPLEXITIES),
        help="Comma-separated complexity values to sample from for each request",
    )
    parser.add_argument(
        "--tool-sequence",
        default=",".join(DEFAULT_TOOL_SEQUENCE),
        help="Comma-separated cyclic tool order, e.g. chat,flashcards,quiz,podcast",
    )
    parser.add_argument(
        "--limit-per-batch",
        type=int,
        default=0,
        help="Optional cap on questions per batch (0 = all available questions)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for complexity selection so runs are reproducible",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan jobs only; do not send requests")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory where log files will be written")
    return parser.parse_args()


def sanitize_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_int_steps(raw_value: str) -> List[int]:
    values: List[int] = []
    for part in str(raw_value or "").split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Stage sizes must be positive integers.")
        values.append(value)
    if not values:
        raise ValueError("At least one stage size must be provided.")
    return values


def parse_text_list(raw_value: str) -> List[str]:
    values = [sanitize_text(part) for part in str(raw_value or "").split(",")]
    values = [value for value in values if value]
    if not values:
        raise ValueError("At least one value must be provided.")
    return values


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
    return extract_questions_from_markdown(Path(config["question_bank_path"]))


def build_base_jobs(limit_per_batch: int) -> List[Dict]:
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


def build_stage_plan(base_jobs: List[Dict], concurrency_steps: List[int]) -> List[List[Dict]]:
    stage_jobs: List[List[Dict]] = []
    cursor = 0

    for stage_concurrency in concurrency_steps:
        next_cursor = cursor + stage_concurrency
        slice_jobs = base_jobs[cursor:next_cursor]
        if len(slice_jobs) < stage_concurrency:
            raise ValueError(
                f"Not enough questions to satisfy stage size {stage_concurrency}. "
                f"Needed {stage_concurrency}, found {len(slice_jobs)}."
            )
        stage_jobs.append(slice_jobs)
        cursor = next_cursor

    return stage_jobs


def ensure_log_dir(path_str: str) -> Path:
    path = Path(path_str).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary(summary_path: Path, summary: Dict):
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_json_request(method: str, url: str, payload: Optional[Dict], timeout_seconds: float) -> Dict:
    request = urllib.request.Request(
        url=url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            data = json.loads(response_text) if response_text else {}
            return {"ok": True, "http_status": response.status, "data": data}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body
        try:
            data = json.loads(body)
            detail = data.get("detail") or body
        except json.JSONDecodeError:
            data = None
        return {"ok": False, "http_status": exc.code, "data": data, "error": sanitize_text(detail)}
    except Exception as exc:
        return {"ok": False, "http_status": None, "data": None, "error": f"{type(exc).__name__}: {exc}"}


def summarize_flashcards(cards) -> Dict:
    card_list = cards if isinstance(cards, list) else []
    preview = ""
    if card_list:
        first_card = card_list[0]
        if isinstance(first_card, dict):
            preview = sanitize_text(first_card.get("question") or first_card.get("front") or "")[:240]
    return {"items_count": len(card_list), "response_preview": preview}


def summarize_quizzes(quizzes) -> Dict:
    quiz_list = quizzes if isinstance(quizzes, list) else []
    preview = ""
    if quiz_list:
        first_quiz = quiz_list[0]
        if isinstance(first_quiz, dict):
            preview = sanitize_text(first_quiz.get("topic") or first_quiz.get("title") or "")[:240]
    return {"items_count": len(quiz_list), "response_preview": preview}


def poll_job(
    *,
    base_url: str,
    job_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> Dict:
    poll_start = time.perf_counter()
    attempts = 0

    while True:
        attempts += 1
        response = make_json_request(
            "GET",
            f"{base_url.rstrip('/')}/jobs/{urllib.parse.quote(job_id)}",
            None,
            timeout_seconds=max(5.0, poll_interval_seconds + 5.0),
        )

        if not response["ok"]:
            response.update({"poll_attempts": attempts, "poll_total_ms": round((time.perf_counter() - poll_start) * 1000, 2)})
            return response

        data = response["data"] or {}
        status = str(data.get("status") or "").lower()
        if status in {"completed", "failed"}:
            response.update(
                {
                    "poll_attempts": attempts,
                    "poll_total_ms": round((time.perf_counter() - poll_start) * 1000, 2),
                    "job_status": status,
                }
            )
            return response

        if (time.perf_counter() - poll_start) >= timeout_seconds:
            return {
                "ok": False,
                "http_status": None,
                "data": data,
                "error": f"Job polling timed out after {timeout_seconds}s",
                "poll_attempts": attempts,
                "poll_total_ms": round((time.perf_counter() - poll_start) * 1000, 2),
                "job_status": status or "processing",
            }

        time.sleep(poll_interval_seconds)


def send_chat_request(job: Dict, args: argparse.Namespace) -> Dict:
    payload = {
        "username": job["username"],
        "active_batch_id": job["batch_id"],
        "message": job["question"],
        "complexity": job["complexity"],
        "tutor_mode": False,
    }
    response = make_json_request("POST", f"{args.base_url.rstrip('/')}/chat", payload, args.timeout)
    result = {
        "endpoint": "/chat",
        "http_status": response["http_status"],
        "server_total_ms": None,
        "response_items_count": 0,
        "response_preview": "",
        "error": response.get("error"),
    }

    if response["ok"]:
        data = response["data"] or {}
        reply = "" if data.get("reply") is None else str(data.get("reply"))
        debug_timings = data.get("debug_timings") if isinstance(data, dict) else None
        result.update(
            {
                "server_total_ms": (debug_timings or {}).get("total_ms") if isinstance(debug_timings, dict) else None,
                "response_items_count": len(data.get("citations") or []),
                "response_preview": sanitize_text(reply)[:240],
                "server_debug_timings": debug_timings,
            }
        )

    return result


def send_flashcards_request(job: Dict, args: argparse.Namespace) -> Dict:
    payload = {
        "username": job["username"],
        "active_batch_id": job["batch_id"],
        "topics": [job["question"]],
        "complexity": job["complexity"],
    }
    response = make_json_request("POST", f"{args.base_url.rstrip('/')}/studio/flashcards", payload, args.timeout)
    result = {
        "endpoint": "/studio/flashcards",
        "http_status": response["http_status"],
        "server_total_ms": None,
        "response_items_count": 0,
        "response_preview": "",
        "error": response.get("error"),
    }

    if response["ok"]:
        data = response["data"] or {}
        summary = summarize_flashcards(data.get("cards"))
        result.update(summary)

    return result


def send_quiz_request(job: Dict, args: argparse.Namespace) -> Dict:
    payload = {
        "username": job["username"],
        "active_batch_id": job["batch_id"],
        "topics": [job["question"]],
        "complexity": job["complexity"],
    }
    response = make_json_request("POST", f"{args.base_url.rstrip('/')}/studio/quiz/generate", payload, args.timeout)
    result = {
        "endpoint": "/studio/quiz/generate",
        "http_status": response["http_status"],
        "server_total_ms": None,
        "response_items_count": 0,
        "response_preview": "",
        "error": response.get("error"),
    }

    if response["ok"]:
        data = response["data"] or {}
        summary = summarize_quizzes(data.get("quizzes"))
        result.update(summary)

    return result


def send_podcast_request(job: Dict, args: argparse.Namespace) -> Dict:
    payload = {
        "username": job["username"],
        "active_batch_id": job["batch_id"],
        "topic": job["question"],
        "complexity": job["complexity"],
    }
    initial = make_json_request("POST", f"{args.base_url.rstrip('/')}/studio/podcast/generate", payload, args.timeout)
    result = {
        "endpoint": "/studio/podcast/generate",
        "http_status": initial["http_status"],
        "server_total_ms": None,
        "response_items_count": 0,
        "response_preview": "",
        "error": initial.get("error"),
        "job_id": None,
        "job_status": None,
        "poll_attempts": 0,
        "poll_total_ms": 0.0,
    }

    if not initial["ok"]:
        return result

    data = initial["data"] or {}
    job_id = data.get("job_id")
    result["job_id"] = job_id

    if not job_id:
        result["error"] = "Podcast generation response missing job_id"
        return result

    poll = poll_job(
        base_url=args.base_url,
        job_id=job_id,
        timeout_seconds=args.job_timeout,
        poll_interval_seconds=args.poll_interval,
    )
    result["poll_attempts"] = poll.get("poll_attempts", 0)
    result["poll_total_ms"] = poll.get("poll_total_ms", 0.0)

    if not poll["ok"]:
        result["error"] = poll.get("error")
        result["job_status"] = poll.get("job_status")
        return result

    poll_data = poll["data"] or {}
    result["job_status"] = poll_data.get("status")
    result["response_preview"] = sanitize_text(poll_data.get("message") or poll_data.get("filename") or "")[:240]
    return result


def execute_job(job: Dict, args: argparse.Namespace) -> Dict:
    started_at = datetime.now(timezone.utc).isoformat()
    perf_start = time.perf_counter()
    base_result = {
        **job,
        "started_at": started_at,
        "status": "failed",
        "http_status": None,
        "client_total_ms": None,
        "server_total_ms": None,
        "response_items_count": 0,
        "response_preview": "",
        "error": None,
    }

    tool = job["tool"]
    if tool == "chat":
        details = send_chat_request(job, args)
    elif tool == "flashcards":
        details = send_flashcards_request(job, args)
    elif tool == "quiz":
        details = send_quiz_request(job, args)
    elif tool == "podcast":
        details = send_podcast_request(job, args)
    else:
        details = {"endpoint": "", "http_status": None, "error": f"Unsupported tool: {tool}"}

    base_result.update(details)
    base_result["client_total_ms"] = round((time.perf_counter() - perf_start) * 1000, 2)

    tool_succeeded = base_result.get("error") is None
    if tool == "podcast" and tool_succeeded:
        tool_succeeded = str(base_result.get("job_status") or "").lower() == "completed"

    base_result["status"] = "success" if tool_succeeded else "failed"
    return base_result


def build_stage_jobs(
    stage_jobs: List[Dict],
    stage_index: int,
    tool_sequence: List[str],
    complexities: List[str],
    rng: random.Random,
) -> List[Dict]:
    planned: List[Dict] = []
    for offset, job in enumerate(stage_jobs):
        planned.append(
            {
                **job,
                "tool": tool_sequence[offset % len(tool_sequence)],
                "complexity": rng.choice(complexities),
                "stage_index": stage_index,
            }
        )
    return planned


def count_by_key(rows: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


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
        f"studio_mixed_load_test_{run_stamp}_stage{stage_index:02d}_concurrency_{stage_concurrency}.jsonl"
    )
    summary_log_path = log_dir / (
        f"studio_mixed_load_test_{run_stamp}_stage{stage_index:02d}_concurrency_{stage_concurrency}_summary.json"
    )
    stage_window_seconds = max(0, int(args.stage_window_seconds))
    launch_interval_seconds = (
        float(stage_window_seconds) / float(max(1, len(jobs) - 1))
        if len(jobs) > 1 and stage_window_seconds > 0
        else 0.0
    )

    print(
        f"\n=== Stage {stage_index}/{stage_count} | requests={len(jobs)} ===\n"
        f"Detailed log file: {detailed_log_path}\n"
        f"Summary log file:  {summary_log_path}\n"
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

                future = executor.submit(execute_job, job, args)
                future_map[future] = job

            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                log_file.flush()

                print(
                    f"[{result['status'].upper()}][{result['tool']}][{result['complexity']}] "
                    f"{result['request_id']} user={result['username']} "
                    f"client_total_ms={result['client_total_ms']} http_status={result['http_status']}"
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
        "timeout_seconds": args.timeout,
        "job_timeout_seconds": args.job_timeout,
        "poll_interval_seconds": args.poll_interval,
        "total_jobs": len(jobs),
        "success_count": len(success_results),
        "failure_count": len(failed_results),
        "avg_client_total_ms": round(sum(client_durations) / len(client_durations), 2) if client_durations else None,
        "max_client_total_ms": round(max(client_durations), 2) if client_durations else None,
        "avg_server_total_ms": round(sum(server_durations) / len(server_durations), 2) if server_durations else None,
        "max_server_total_ms": round(max(server_durations), 2) if server_durations else None,
        "tool_counts": count_by_key(jobs, "tool"),
        "complexity_counts": count_by_key(jobs, "complexity"),
        "detailed_log_file": str(detailed_log_path),
        "manifest": manifest,
        "failures": [
            {
                "request_id": item["request_id"],
                "tool": item["tool"],
                "batch_id": item["batch_id"],
                "username": item["username"],
                "complexity": item["complexity"],
                "http_status": item["http_status"],
                "error": item["error"],
                "job_status": item.get("job_status"),
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
    concurrency_steps = parse_int_steps(args.concurrency_steps)
    complexities = parse_text_list(args.complexities)
    tool_sequence = parse_text_list(args.tool_sequence)
    invalid_tools = [tool for tool in tool_sequence if tool not in {"chat", "flashcards", "quiz", "podcast"}]
    if invalid_tools:
        raise ValueError(f"Unsupported tool names: {invalid_tools}")

    base_jobs = build_base_jobs(args.limit_per_batch)
    stage_plan = build_stage_plan(base_jobs, concurrency_steps)
    log_dir = ensure_log_dir(args.log_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = random.Random(args.seed)

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

    print("Studio Mixed Load Test Manifest")
    print(json.dumps(manifest, indent=2))
    print(f"Concurrency stages: {concurrency_steps}")
    print(f"Stage window seconds: {args.stage_window_seconds}")
    print(f"Tool sequence: {tool_sequence}")
    print(f"Complexities: {complexities}")
    print(f"Random seed: {args.seed}")

    stage_job_counts = [len(stage_jobs) for stage_jobs in stage_plan]
    planned_stage_jobs = [
        build_stage_jobs(stage_jobs, stage_index, tool_sequence, complexities, rng)
        for stage_index, stage_jobs in enumerate(stage_plan, start=1)
    ]

    if args.dry_run:
        summary_log_path = log_dir / f"studio_mixed_load_test_{run_stamp}_dry_run_summary.json"
        summary = {
            "mode": "dry_run",
            "base_url": args.base_url,
            "concurrency_steps": concurrency_steps,
            "stage_window_seconds": args.stage_window_seconds,
            "timeout_seconds": args.timeout,
            "job_timeout_seconds": args.job_timeout,
            "poll_interval_seconds": args.poll_interval,
            "seed": args.seed,
            "tool_sequence": tool_sequence,
            "complexities": complexities,
            "total_jobs_available": len(base_jobs),
            "total_jobs_planned": sum(stage_job_counts),
            "stage_request_counts": stage_job_counts,
            "planned_tool_counts": [count_by_key(stage_jobs, "tool") for stage_jobs in planned_stage_jobs],
            "planned_complexity_counts": [count_by_key(stage_jobs, "complexity") for stage_jobs in planned_stage_jobs],
            "manifest": manifest,
        }
        write_summary(summary_log_path, summary)
        print("Dry run completed. No requests were sent.")
        return

    all_stage_summaries = []
    for stage_index, (stage_concurrency, stage_jobs) in enumerate(zip(concurrency_steps, planned_stage_jobs), start=1):
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

    aggregate_summary_path = log_dir / f"studio_mixed_load_test_{run_stamp}_all_stages_summary.json"
    aggregate_summary = {
        "mode": "multi_stage_execute",
        "base_url": args.base_url,
        "concurrency_steps": concurrency_steps,
        "stage_window_seconds": args.stage_window_seconds,
        "timeout_seconds": args.timeout,
        "job_timeout_seconds": args.job_timeout,
        "poll_interval_seconds": args.poll_interval,
        "seed": args.seed,
        "tool_sequence": tool_sequence,
        "complexities": complexities,
        "total_jobs_available": len(base_jobs),
        "total_jobs_planned": sum(stage_job_counts),
        "stage_request_counts": stage_job_counts,
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
