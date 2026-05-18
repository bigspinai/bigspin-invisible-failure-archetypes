#!/usr/bin/env python3
"""
Batch scoring pipeline for the two-pass (K-IND+R4) tagging architecture.

Submits batch jobs to both Anthropic (Opus 4.6) and OpenAI (GPT-5.4) APIs.
Each turn generates TWO requests (naive pass + calibrated pass), which are
merged after completion.

Pipeline stages:
    prepare   — Build batch request files from a turn-pair JSONL dataset
    submit    — Submit batch files to APIs (5k requests per batch)
    status    — Check status of submitted batches
    download  — Download completed batch results
    merge     — Merge 2-pass results and reassemble into transcript-level output

Usage:
    # Full pipeline
    python batch_score.py prepare --dataset data/datasets/test_1k.jsonl
    python batch_score.py submit  --dataset data/datasets/test_1k.jsonl
    python batch_score.py status  --dataset data/datasets/test_1k.jsonl
    python batch_score.py download --dataset data/datasets/test_1k.jsonl
    python batch_score.py merge   --dataset data/datasets/test_1k.jsonl

    # Dry run (prepare only, show stats)
    python batch_score.py prepare --dataset data/datasets/test_1k.jsonl --dry-run
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Annotator configs ──────────────────────────────────────────────────
ANNOTATORS = {
    "opus": {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "label": "Opus 4.6",
    },
    "sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "label": "Sonnet 4.6",
    },
    "gpt5": {
        "provider": "openai",
        "model": "gpt-5.4",
        "label": "GPT-5.4",
    },
}

BATCH_SIZE = 2000  # Both providers have ~200-256MB file size limits (~58KB/req avg)

TASKS = ("ai", "user")  # Supported tagging tasks


def active_annotators(args) -> list[str]:
    """Return annotator keys to process, respecting --annotator filter."""
    if getattr(args, "annotator", None):
        return args.annotator
    return list(ANNOTATORS.keys())

# ── Prompt building ────────────────────────────────────────────────────

_builder_naive = None
_builder_cal = None


def get_builders():
    """Lazy-init prompt builders."""
    global _builder_naive, _builder_cal
    if _builder_naive is None:
        from prompt_builder import PromptBuilder
        _builder_cal = PromptBuilder.from_files(
            SCRIPT_DIR / "taxonomy.json",
            SCRIPT_DIR / "calibration.json",
        )
        _builder_naive = PromptBuilder.from_files(
            SCRIPT_DIR / "taxonomy.json",
            SCRIPT_DIR / "calibration.json",
            naive=True,
        )
    return _builder_naive, _builder_cal


def build_requests_for_turn(turn: dict, annotator_key: str, task: str = "ai") -> list[dict]:
    """Build batch request dicts for a single turn.

    For task="ai": 2 requests (naive + calibrated) — two-pass architecture.
    For task="user": 1 request (single-pass, no calibration needed).
    """
    builder_naive, builder_cal = get_builders()
    annotator = ANNOTATORS[annotator_key]

    cid = turn["conversation_id"]
    tn = turn["turn_number"]
    base_id = f"{cid}_{tn}"

    if task == "user":
        # Single-pass: user signals have no calibration
        user_prompt = builder_cal.build_user_prompt(turn)
        return [
            {
                "custom_id": f"{base_id}_user",
                "provider": annotator["provider"],
                "model": annotator["model"],
                "prompt": user_prompt,
                "system": "",
                "max_tokens": 4096,
                "pass_type": "user",
            },
        ]

    # task == "ai": two-pass architecture
    # Pass 1: naive (self-evident signals)
    naive_prompt = builder_naive.build_naive_ai_prompt(turn)

    # Pass 2: calibrated (structured prompt with phases)
    cal_prompt = builder_cal.build_ai_prompt(turn)

    return [
        {
            "custom_id": f"{base_id}_naive",
            "provider": annotator["provider"],
            "model": annotator["model"],
            "prompt": naive_prompt,
            "system": "",
            "max_tokens": 4096,
            "pass_type": "naive",
        },
        {
            "custom_id": f"{base_id}_calibrated",
            "provider": annotator["provider"],
            "model": annotator["model"],
            "prompt": cal_prompt,
            "system": "",
            "max_tokens": 4096,
            "pass_type": "calibrated",
        },
    ]


# ── Batch file formatting (provider-specific) ─────────────────────────

def format_anthropic_request(req: dict) -> dict:
    """Format a request for Anthropic's Message Batches API."""
    return {
        "custom_id": req["custom_id"],
        "params": {
            "model": req["model"],
            "max_tokens": req["max_tokens"],
            "temperature": 0,
            "messages": [{"role": "user", "content": req["prompt"]}],
        },
    }


def format_openai_request(req: dict) -> dict:
    """Format a request for OpenAI's Batch API."""
    messages = []
    if req.get("system"):
        messages.append({"role": "system", "content": req["system"]})
    messages.append({"role": "user", "content": req["prompt"]})

    return {
        "custom_id": req["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": req["model"],
            "max_completion_tokens": req["max_tokens"],
            "temperature": 0,
            "messages": messages,
        },
    }


# ── Job directory structure ────────────────────────────────────────────

def get_job_dir(dataset_path: Path) -> Path:
    """Get/create the batch job directory for a dataset."""
    dataset_name = dataset_path.stem  # e.g. "test_1k"
    job_dir = SCRIPT_DIR / "data" / "batch_jobs" / dataset_name
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def batch_prefix(annotator_key: str, task: str) -> str:
    """Return the file prefix for batch files: e.g. 'opus' or 'opus_user'."""
    if task == "user":
        return f"{annotator_key}_user"
    return annotator_key


# ── Stage: prepare ─────────────────────────────────────────────────────

def cmd_prepare(args):
    """Build batch request JSONL files from a dataset.

    Streams turns through prompt building one at a time to avoid OOM
    on large datasets (each prompt can be 20-80KB).
    """
    dataset_path = Path(args.dataset)
    task = args.task
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    reqs_per_turn = 1 if task == "user" else 2
    task_label = "USER SIGNALS (single-pass)" if task == "user" else "AI SIGNALS (two-pass)"

    print("=" * 70)
    print(f"PREPARE BATCH REQUESTS — {task_label}")
    print("=" * 70)

    # Count turns without loading all into memory
    n_turns = sum(1 for _ in open(dataset_path))
    print(f"  Dataset: {dataset_path.name} ({n_turns} turns)")

    job_dir = get_job_dir(dataset_path)

    for annotator_key in active_annotators(args):
        annotator = ANNOTATORS[annotator_key]
        provider = annotator["provider"]
        formatter = (format_anthropic_request if provider == "anthropic"
                     else format_openai_request)
        prefix = batch_prefix(annotator_key, task)

        print(f"\n  Building requests for {annotator['label']}...")

        total_requests = n_turns * reqs_per_turn
        n_batches = (total_requests + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    Total requests: {total_requests} ({n_turns} turns × {reqs_per_turn} pass{'es' if reqs_per_turn > 1 else ''})")
        print(f"    Batches: {n_batches} (max {BATCH_SIZE} per batch)")

        if args.dry_run:
            # Build one sample to show stats
            with open(dataset_path) as f:
                sample_turn = json.loads(f.readline())
            reqs = build_requests_for_turn(sample_turn, annotator_key, task=task)
            sample = reqs[0]
            print(f"    Sample request (custom_id={sample['custom_id']}):")
            print(f"      provider: {sample['provider']}")
            print(f"      model: {sample['model']}")
            print(f"      pass_type: {sample['pass_type']}")
            print(f"      prompt length: {len(sample['prompt']):,} chars")
            continue

        # Stream: read turns one at a time, write formatted requests to batch files
        batch_idx = 0
        batch_count = 0
        batch_file = job_dir / f"{prefix}_batch_{batch_idx:03d}.jsonl"
        out_f = open(batch_file, "w")
        written_total = 0

        with open(dataset_path) as f:
            for i, line in enumerate(f):
                turn = json.loads(line)
                reqs = build_requests_for_turn(turn, annotator_key, task=task)

                for req in reqs:
                    formatted = formatter(req)
                    out_f.write(json.dumps(formatted) + "\n")
                    batch_count += 1
                    written_total += 1

                    # Rotate batch file if needed
                    if batch_count >= BATCH_SIZE:
                        out_f.close()
                        print(f"    Wrote {batch_file.name} ({batch_count} requests)")
                        batch_idx += 1
                        batch_count = 0
                        batch_file = job_dir / f"{prefix}_batch_{batch_idx:03d}.jsonl"
                        out_f = open(batch_file, "w")

                # Progress
                if (i + 1) % 2000 == 0:
                    print(f"      [{i+1}/{n_turns}] turns processed...")

        # Close final batch file
        out_f.close()
        if batch_count > 0:
            print(f"    Wrote {batch_file.name} ({batch_count} requests)")
        else:
            # Empty final file — remove it
            batch_file.unlink(missing_ok=True)

        print(f"    Total written: {written_total} requests")

    if args.dry_run:
        total_reqs = n_turns * reqs_per_turn * len(ANNOTATORS)
        print(f"\n  DRY RUN — would create {total_reqs} requests across {len(ANNOTATORS)} annotators")
        return

    # Write job manifest
    manifest_name = "manifest.json" if task == "ai" else f"manifest_{task}.json"
    manifest = {
        "dataset": str(dataset_path),
        "task": task,
        "n_turns": n_turns,
        "annotators": list(ANNOTATORS.keys()),
        "batch_size": BATCH_SIZE,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
    }
    with open(job_dir / manifest_name, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Job directory: {job_dir}")
    print(f"  Next: python batch_score.py submit --task {task} --dataset {dataset_path}")


# ── Stage: submit ──────────────────────────────────────────────────────

def cmd_submit(args):
    """Submit batch files to APIs."""
    dataset_path = Path(args.dataset)
    task = args.task
    job_dir = get_job_dir(dataset_path)

    manifest_name = "manifest.json" if task == "ai" else f"manifest_{task}.json"
    if not (job_dir / manifest_name).exists():
        print(f"ERROR: Run 'prepare --task {task}' first")
        sys.exit(1)

    task_label = "USER SIGNALS" if task == "user" else "AI SIGNALS"
    print("=" * 70)
    print(f"SUBMIT BATCH JOBS — {task_label}")
    print("=" * 70)

    submissions = {}

    for annotator_key in active_annotators(args):
        annotator = ANNOTATORS[annotator_key]
        provider = annotator["provider"]
        prefix = batch_prefix(annotator_key, task)

        # Find batch files for this annotator+task
        batch_files = sorted(job_dir.glob(f"{prefix}_batch_*.jsonl"))
        if not batch_files:
            print(f"  {annotator_key}: no batch files found")
            continue

        print(f"\n  {annotator['label']} ({provider}): {len(batch_files)} batch(es)")

        for batch_file in batch_files:
            batch_name = batch_file.stem

            # Skip if already submitted
            status_file = job_dir / f"{batch_name}.status.json"
            if status_file.exists():
                existing = json.loads(status_file.read_text())
                if existing.get("batch_id"):
                    print(f"    {batch_name}: already submitted (id={existing['batch_id']})")
                    submissions[batch_name] = existing
                    continue

            n_lines = sum(1 for _ in open(batch_file))
            print(f"    Submitting {batch_name} ({n_lines} requests)...")

            try:
                if provider == "anthropic":
                    batch_id = _submit_anthropic_batch(batch_file)
                else:
                    batch_id = _submit_openai_batch(batch_file)

                status = {
                    "batch_name": batch_name,
                    "batch_id": batch_id,
                    "provider": provider,
                    "annotator": annotator_key,
                    "n_requests": n_lines,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "status": "submitted",
                }
                with open(status_file, "w") as f:
                    json.dump(status, f, indent=2)

                submissions[batch_name] = status
                print(f"    → batch_id: {batch_id}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    # Update manifest
    manifest_path = job_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "submitted"
    manifest["submissions"] = submissions
    manifest["submitted_at"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Next: python batch_score.py status --dataset {dataset_path}")


def _submit_anthropic_batch(batch_file: Path) -> str:
    """Submit a batch to Anthropic's Message Batches API. Returns batch_id."""
    from anthropic import Anthropic
    client = Anthropic()

    # Read requests from file
    requests = []
    with open(batch_file) as f:
        for line in f:
            requests.append(json.loads(line))

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def _submit_openai_batch(batch_file: Path) -> str:
    """Submit a batch to OpenAI's Batch API. Returns batch_id."""
    from openai import OpenAI
    client = OpenAI()

    # Upload the batch file
    with open(batch_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    # Create the batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


# ── Stage: status ──────────────────────────────────────────────────────

def resolve_tasks(task: str) -> list[str]:
    """Expand 'all' into the individual task list."""
    if task == "all":
        return list(TASKS)
    return [task]


def _filter_status_files(job_dir: Path, task: str, args) -> list[Path]:
    """Get status files filtered by task and annotator."""
    if task == "user":
        status_files = sorted(job_dir.glob("*_user_batch_*.status.json"))
    else:
        status_files = sorted(
            sf for sf in job_dir.glob("*.status.json")
            if "_user_" not in sf.name
        )
    # Filter by annotator if specified
    ann_keys = active_annotators(args)
    if getattr(args, "annotator", None):
        prefixes = [batch_prefix(k, task) for k in ann_keys]
        status_files = [sf for sf in status_files
                        if any(sf.name.startswith(p + "_batch_") for p in prefixes)]
    return status_files


def cmd_status(args):
    """Check status of submitted batches."""
    dataset_path = Path(args.dataset)
    job_dir = get_job_dir(dataset_path)

    all_done = True
    any_found = False

    for task in resolve_tasks(args.task):
        task_label = "USER SIGNALS" if task == "user" else "AI SIGNALS"
        status_files = _filter_status_files(job_dir, task, args)
        if not status_files:
            continue
        any_found = True

        print("=" * 70)
        print(f"BATCH STATUS — {task_label}")
        print("=" * 70)

        for sf in status_files:
            status = json.loads(sf.read_text())
            batch_id = status.get("batch_id")
            provider = status.get("provider")
            batch_name = status.get("batch_name")

            if not batch_id:
                print(f"  {batch_name}: no batch_id")
                all_done = False
                continue

            try:
                if provider == "anthropic":
                    current = _check_anthropic_batch(batch_id)
                else:
                    current = _check_openai_batch(batch_id)

                status.update(current)
                status["checked_at"] = datetime.now(timezone.utc).isoformat()
                with open(sf, "w") as f:
                    json.dump(status, f, indent=2)

                state = current.get("status", "unknown")
                extra = ""
                if current.get("completed_count") is not None:
                    extra = f" ({current['completed_count']}/{current['total_count']})"
                print(f"  {batch_name}: {state}{extra}")

                if state not in ("ended", "completed"):
                    all_done = False

            except Exception as e:
                print(f"  {batch_name}: ERROR checking — {e}")
                all_done = False

    if not any_found:
        print("  No batches found. Run 'submit' first.")
        return

    if all_done:
        print(f"\n  All batches complete!")
        print(f"  Next: python batch_score.py download --dataset {dataset_path}")
    else:
        print(f"\n  Some batches still running. Check again later.")


def _check_anthropic_batch(batch_id: str) -> dict:
    """Check Anthropic batch status."""
    from anthropic import Anthropic
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    return {
        "status": batch.processing_status,
        "completed_count": batch.request_counts.succeeded if hasattr(batch, 'request_counts') else None,
        "failed_count": batch.request_counts.errored if hasattr(batch, 'request_counts') else None,
        "total_count": (batch.request_counts.succeeded + batch.request_counts.processing +
                        batch.request_counts.errored + batch.request_counts.canceled)
        if hasattr(batch, 'request_counts') else None,
        "results_url": batch.results_url if hasattr(batch, 'results_url') else None,
    }


def _check_openai_batch(batch_id: str) -> dict:
    """Check OpenAI batch status."""
    from openai import OpenAI
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    return {
        "status": batch.status,
        "completed_count": counts.completed if counts else None,
        "failed_count": counts.failed if counts else None,
        "total_count": counts.total if counts else None,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


# ── Stage: download ────────────────────────────────────────────────────

def cmd_download(args):
    """Download completed batch results."""
    dataset_path = Path(args.dataset)
    job_dir = get_job_dir(dataset_path)
    results_dir = job_dir / "raw_results"
    results_dir.mkdir(exist_ok=True)

    for task in resolve_tasks(args.task):
        task_label = "USER SIGNALS" if task == "user" else "AI SIGNALS"
        status_files = _filter_status_files(job_dir, task, args)
        if not status_files:
            continue

        print("=" * 70)
        print(f"DOWNLOAD BATCH RESULTS — {task_label}")
        print("=" * 70)

        for sf in status_files:
            status = json.loads(sf.read_text())
            batch_id = status.get("batch_id")
            provider = status.get("provider")
            batch_name = status.get("batch_name")
            state = status.get("status")

            result_file = results_dir / f"{batch_name}.results.jsonl"
            if result_file.exists():
                n = sum(1 for _ in open(result_file))
                print(f"  {batch_name}: already downloaded ({n} results)")
                continue

            if state not in ("ended", "completed"):
                print(f"  {batch_name}: not ready (status={state})")
                continue

            print(f"  Downloading {batch_name}...")

            try:
                if provider == "anthropic":
                    results = _download_anthropic_results(batch_id)
                else:
                    results = _download_openai_results(status)

                with open(result_file, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                print(f"    → {result_file.name} ({len(results)} results)")

            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"\n  Next: python batch_score.py merge --dataset {dataset_path}")


def _download_anthropic_results(batch_id: str) -> list[dict]:
    """Download results from Anthropic Message Batches API."""
    from anthropic import Anthropic
    client = Anthropic()

    results = []
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            content = result.result.message.content
            if content and len(content) > 0:
                text = content[0].text
            else:
                text = ""
            results.append({
                "custom_id": custom_id,
                "status": "ok" if text else "error",
                "response": text,
                **({"error": "empty content"} if not text else {}),
            })
        else:
            results.append({
                "custom_id": custom_id,
                "status": "error",
                "error": str(result.result),
            })
    return results


def _download_openai_results(status: dict) -> list[dict]:
    """Download results from OpenAI Batch API."""
    from openai import OpenAI
    client = OpenAI()

    output_file_id = status.get("output_file_id")
    if not output_file_id:
        raise ValueError("No output_file_id in status")

    content = client.files.content(output_file_id)
    text = content.text

    results = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record["custom_id"]
        response_body = record.get("response", {}).get("body", {})

        if record.get("error"):
            results.append({
                "custom_id": custom_id,
                "status": "error",
                "error": str(record["error"]),
            })
        else:
            choices = response_body.get("choices", [])
            text_content = choices[0]["message"]["content"] if choices else ""
            results.append({
                "custom_id": custom_id,
                "status": "ok",
                "response": text_content,
            })
    return results


# ── Stage: merge ───────────────────────────────────────────────────────

def cmd_merge(args):
    """Merge results and reassemble unified transcript-level output.

    Processes both AI and user signals, then joins them into unified turn/transcript
    files where each turn has both ai_signals and user_signals.

    AI signals come in two variants:
      - baseline: all signals from naive pass (full taxonomy)
      - calibrated: cherry-pick PASS1 from naive, PASS2 from calibrated
    """
    dataset_path = Path(args.dataset)
    job_dir = get_job_dir(dataset_path)
    results_dir = job_dir / "raw_results"
    output_dir = SCRIPT_DIR / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    from api_client import parse_json_response

    for annotator_key in active_annotators(args):
        print("=" * 70)
        print(f"MERGE — {ANNOTATORS[annotator_key]['label']}")
        print("=" * 70)

        # ── AI signals (two-pass) ────────────────────────────────────
        ai_prefix = batch_prefix(annotator_key, "ai")
        ai_result_files = sorted(results_dir.glob(f"{ai_prefix}_batch_*.results.jsonl"))

        baseline_by_key = {}   # (cid, tn) -> turn dict
        calibrated_by_key = {}

        if ai_result_files:
            print(f"\n  AI signals:")
            baseline_turns, calibrated_turns = _merge_ai_signals(ai_result_files, annotator_key)
            for t in baseline_turns:
                baseline_by_key[(t["conversation_id"], t["turn_number"])] = t
            for t in calibrated_turns:
                calibrated_by_key[(t["conversation_id"], t["turn_number"])] = t
        else:
            print(f"\n  AI signals: no result files found")

        # ── User signals (single-pass or borrowed) ────────────────
        user_by_key = {}  # (cid, tn) -> turn dict

        user_source = getattr(args, "user_signals_from", None)
        if user_source:
            # Borrow user signals from an existing dataset's merged results,
            # but only for conversation IDs that appear in our AI results
            ai_cids = set()
            for key in (set(baseline_by_key.keys()) | set(calibrated_by_key.keys())):
                ai_cids.add(key[0])

            print(f"\n  User signals: borrowing from {user_source} (filtered to {len(ai_cids)} AI conversation IDs)")
            for scheme_name in ("calibrated", "baseline"):
                donor_path = output_dir / f"{user_source}_{annotator_key}_{scheme_name}_turns.jsonl"
                if donor_path.exists():
                    n_borrowed = 0
                    n_skipped = 0
                    with open(donor_path) as f:
                        for line in f:
                            r = json.loads(line)
                            if r["conversation_id"] not in ai_cids:
                                n_skipped += 1
                                continue
                            key = (r["conversation_id"], r["turn_number"])
                            if key not in user_by_key and r.get("user_signals"):
                                user_by_key[key] = {
                                    "conversation_id": r["conversation_id"],
                                    "turn_number": r["turn_number"],
                                    "annotator": annotator_key,
                                    "user_signals": r["user_signals"],
                                    "signal_count": r.get("user_signal_count", len(r["user_signals"])),
                                    "complete": True,
                                }
                                n_borrowed += 1
                    print(f"    Loaded {n_borrowed} turns with user signals from {donor_path.name} ({n_skipped} outside dataset skipped)")
                    break  # prefer calibrated, fall back to baseline
            if not user_by_key:
                print(f"    WARNING: No user signals found in {user_source} for {annotator_key}")
        else:
            user_prefix = batch_prefix(annotator_key, "user")
            user_result_files = sorted(results_dir.glob(f"{user_prefix}_batch_*.results.jsonl"))

            if user_result_files:
                print(f"\n  User signals:")
                user_turns = _merge_user_signals(user_result_files, annotator_key)
                for t in user_turns:
                    user_by_key[(t["conversation_id"], t["turn_number"])] = t
            else:
                print(f"\n  User signals: no result files found")

        # ── Join into unified turns ──────────────────────────────────
        all_keys = sorted(set(baseline_by_key.keys()) | set(calibrated_by_key.keys()) | set(user_by_key.keys()))

        if not all_keys:
            print(f"\n  No results to merge for {annotator_key}")
            continue

        for scheme, ai_by_key in [("baseline", baseline_by_key), ("calibrated", calibrated_by_key)]:
            unified_turns = []
            for key in all_keys:
                cid, tn = key
                ai_turn = ai_by_key.get(key)
                user_turn = user_by_key.get(key)

                unified = {
                    "conversation_id": cid,
                    "turn_number": tn,
                    "annotator": annotator_key,
                    "scheme": scheme,
                    "ai_signals": ai_turn["ai_signals"] if ai_turn else [],
                    "user_signals": user_turn["user_signals"] if user_turn else [],
                    "ai_signal_count": ai_turn["signal_count"] if ai_turn else 0,
                    "user_signal_count": user_turn["signal_count"] if user_turn else 0,
                    "complete": bool(ai_turn and user_turn and ai_turn.get("complete", False) and user_turn.get("complete", False)),
                }
                unified_turns.append(unified)

            _write_unified_outputs(unified_turns, dataset_path, annotator_key, scheme, output_dir)

    print(f"\n{'=' * 70}")
    print("DONE — Results in data/results/")
    print(f"{'=' * 70}")


def _write_unified_outputs(
    unified_turns: list[dict],
    dataset_path: Path,
    annotator_key: str,
    scheme: str,
    output_dir: Path,
):
    """Write unified turn-level and transcript-level JSONL files."""
    from collections import defaultdict, Counter

    # Turn-level
    turn_output = output_dir / f"{dataset_path.stem}_{annotator_key}_{scheme}_turns.jsonl"
    with open(turn_output, "w") as f:
        for t in sorted(unified_turns, key=lambda x: (x["conversation_id"], x["turn_number"])):
            f.write(json.dumps(t) + "\n")
    n_complete = sum(1 for t in unified_turns if t["complete"])
    print(f"    {scheme}_turns: {turn_output.name} ({len(unified_turns)} turns, {n_complete} complete)")

    # Transcript-level
    by_conv = defaultdict(list)
    for t in unified_turns:
        by_conv[t["conversation_id"]].append(t)

    transcript_output = output_dir / f"{dataset_path.stem}_{annotator_key}_{scheme}_transcripts.jsonl"
    with open(transcript_output, "w") as f:
        for cid in sorted(by_conv.keys()):
            conv_turns = sorted(by_conv[cid], key=lambda x: x["turn_number"])

            all_ai_signals = []
            all_user_signals = []
            for t in conv_turns:
                for sig in t["ai_signals"]:
                    all_ai_signals.append({**sig, "turn_number": t["turn_number"]})
                for sig in t["user_signals"]:
                    all_user_signals.append({**sig, "turn_number": t["turn_number"]})

            ai_counts = Counter(s["signal"] for s in all_ai_signals)
            user_counts = Counter(s["signal"] for s in all_user_signals)

            transcript_rec = {
                "conversation_id": cid,
                "annotator": annotator_key,
                "scheme": scheme,
                "n_turns": len(conv_turns),
                "turns": conv_turns,
                "ai_signal_summary": dict(ai_counts),
                "user_signal_summary": dict(user_counts),
                "total_ai_signals": len(all_ai_signals),
                "total_user_signals": len(all_user_signals),
                "all_complete": all(t["complete"] for t in conv_turns),
            }
            f.write(json.dumps(transcript_rec) + "\n")

    n_convos = len(by_conv)
    n_all_complete = sum(1 for cid in by_conv if all(t["complete"] for t in by_conv[cid]))
    print(f"    {scheme}_transcripts: {transcript_output.name} ({n_convos} conversations, {n_all_complete} fully complete)")


def _parse_ai_pairs(result_files: list):
    """Parse raw two-pass AI results into {base_id: {naive: raw, calibrated: raw}}.

    Also extracts parsed signal lists per pass. Returns (pairs, n_ok, n_err)
    where pairs[base_id] = {
        "naive_raw": str|None, "calibrated_raw": str|None,
        "naive_signals": list[dict], "calibrated_signals": list[dict],
    }
    """
    from api_client import parse_json_response

    pairs = {}
    n_ok = 0
    n_err = 0
    for rf in result_files:
        with open(rf) as f:
            for line in f:
                r = json.loads(line)
                custom_id = r["custom_id"]
                parts = custom_id.rsplit("_", 1)
                base_id = parts[0]
                pass_type = parts[1]

                if base_id not in pairs:
                    pairs[base_id] = {
                        "naive_raw": None, "calibrated_raw": None,
                        "naive_signals": [], "calibrated_signals": [],
                    }

                if r["status"] == "ok":
                    pairs[base_id][f"{pass_type}_raw"] = r["response"]
                    n_ok += 1

                    parsed = parse_json_response(r["response"])
                    if parsed:
                        signals = []
                        for s in (parsed.get("signals", [])
                                  + parsed.get("phase1_signals", [])
                                  + parsed.get("phase2_signals", [])):
                            if isinstance(s, dict) and s.get("signal"):
                                signals.append(s)
                        pairs[base_id][f"{pass_type}_signals"] = signals
                else:
                    n_err += 1

    return pairs, n_ok, n_err


def _merge_ai_signals(result_files: list, annotator_key: str) -> tuple[list[dict], list[dict]]:
    """Merge two-pass AI signal results into baseline and calibrated variants.

    Returns (baseline_turns, calibrated_turns) where:
      - baseline: all signals from naive pass (full taxonomy, no cherry-pick)
      - calibrated: PASS1 signals from naive + PASS2 signals from calibrated
    """
    from signal_groups import PASS1_SIGNALS, PASS2_SIGNALS
    ALL_TAXONOMY = PASS1_SIGNALS | PASS2_SIGNALS

    pairs, n_ok, n_err = _parse_ai_pairs(result_files)
    print(f"    Raw results: {n_ok} ok, {n_err} errors")

    baseline_turns = []
    calibrated_turns = []
    n_complete = 0
    n_partial = 0

    for base_id, pd in sorted(pairs.items()):
        last_underscore = base_id.rfind("_")
        cid = int(base_id[:last_underscore]) if base_id[:last_underscore].isdigit() else base_id[:last_underscore]
        tn = int(base_id[last_underscore + 1:])

        naive_signals = pd["naive_signals"]
        cal_signals = pd["calibrated_signals"]

        has_both = pd["naive_raw"] is not None and pd["calibrated_raw"] is not None
        if has_both:
            n_complete += 1
        else:
            n_partial += 1

        # Baseline: all taxonomy signals from naive pass (filtered to canonical names)
        baseline_filtered = [s for s in naive_signals if s.get("signal") in ALL_TAXONOMY]
        baseline_turns.append({
            "conversation_id": cid,
            "turn_number": tn,
            "annotator": annotator_key,
            "ai_signals": baseline_filtered,
            "signal_count": len(baseline_filtered),
            "scheme": "baseline",
            "complete": pd["naive_raw"] is not None,
        })

        # Calibrated: cherry-pick PASS1 from naive, PASS2 from calibrated
        cherry_picked = []
        seen = set()
        for s in naive_signals:
            sig_name = s.get("signal")
            if sig_name in PASS1_SIGNALS and sig_name not in seen:
                cherry_picked.append(s)
                seen.add(sig_name)
        for s in cal_signals:
            sig_name = s.get("signal")
            if sig_name in PASS2_SIGNALS and sig_name not in seen:
                cherry_picked.append(s)
                seen.add(sig_name)

        calibrated_turns.append({
            "conversation_id": cid,
            "turn_number": tn,
            "annotator": annotator_key,
            "ai_signals": cherry_picked,
            "signal_count": len(cherry_picked),
            "pass1_count": len(naive_signals),
            "pass2_count": len(cal_signals),
            "scheme": "calibrated",
            "complete": has_both,
        })

    print(f"    Merged: {n_complete} complete, {n_partial} partial")
    return baseline_turns, calibrated_turns


def _merge_user_signals(result_files: list, annotator_key: str) -> list[dict]:
    """Merge single-pass user signal results."""
    from api_client import parse_json_response

    n_ok = 0
    n_err = 0
    merged_turns = []

    for rf in result_files:
        with open(rf) as f:
            for line in f:
                r = json.loads(line)
                custom_id = r["custom_id"]
                # custom_id: "{cid}_{tn}_user"
                parts = custom_id.rsplit("_", 1)
                base_id = parts[0]

                last_underscore = base_id.rfind("_")
                cid = int(base_id[:last_underscore]) if base_id[:last_underscore].isdigit() else base_id[:last_underscore]
                tn = int(base_id[last_underscore + 1:])

                if r["status"] == "ok":
                    n_ok += 1
                    signals = []
                    parsed = parse_json_response(r["response"])
                    if parsed:
                        for s in parsed.get("signals", []) + parsed.get("phase1_signals", []) + parsed.get("phase2_signals", []):
                            if isinstance(s, dict) and s.get("signal"):
                                signals.append(s)
                else:
                    n_err += 1
                    signals = []

                merged_turns.append({
                    "conversation_id": cid,
                    "turn_number": tn,
                    "annotator": annotator_key,
                    "user_signals": signals,
                    "signal_count": len(signals),
                    "complete": r["status"] == "ok",
                })

    print(f"    Raw results: {n_ok} ok, {n_err} errors")
    return merged_turns


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch scoring pipeline (two-pass K-IND+R4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tasks:
    --task ai     AI signals, two-pass K-IND+R4 architecture (default)
    --task user   User signals, single-pass (no calibration)

Example workflow (AI signals):
    python batch_score.py prepare  --dataset data/datasets/test_1k.jsonl
    python batch_score.py submit   --dataset data/datasets/test_1k.jsonl
    python batch_score.py status   --dataset data/datasets/test_1k.jsonl
    python batch_score.py download --dataset data/datasets/test_1k.jsonl
    python batch_score.py merge    --dataset data/datasets/test_1k.jsonl

Example workflow (user signals):
    python batch_score.py prepare  --task user --dataset data/datasets/test_1k.jsonl
    python batch_score.py submit   --task user --dataset data/datasets/test_1k.jsonl
    python batch_score.py status   --task user --dataset data/datasets/test_1k.jsonl
    python batch_score.py download --task user --dataset data/datasets/test_1k.jsonl
    python batch_score.py merge    --task user --dataset data/datasets/test_1k.jsonl
""",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    TASKS_ALL = (*TASKS, "all")

    def add_common_args(p, default_task="ai"):
        p.add_argument("--dataset", required=True, help="Path to turn-pair JSONL dataset")
        p.add_argument("--task", choices=TASKS_ALL if default_task == "all" else TASKS,
                       default=default_task,
                       help="Tagging task: 'ai', 'user', or 'all' (default: %(default)s)")
        p.add_argument("--annotator", choices=list(ANNOTATORS.keys()), nargs="+",
                       help="Run only for these annotators (default: all)")

    # prepare / submit: explicit task required (no "all" default)
    p_prep = subparsers.add_parser("prepare", help="Build batch request files")
    add_common_args(p_prep, default_task="ai")
    p_prep.add_argument("--dry-run", action="store_true")

    p_sub = subparsers.add_parser("submit", help="Submit batches to APIs")
    add_common_args(p_sub, default_task="ai")

    # status / download / merge: default to "all" tasks
    p_stat = subparsers.add_parser("status", help="Check batch status")
    add_common_args(p_stat, default_task="all")

    p_dl = subparsers.add_parser("download", help="Download completed results")
    add_common_args(p_dl, default_task="all")

    p_merge = subparsers.add_parser("merge", help="Merge results")
    add_common_args(p_merge, default_task="all")
    p_merge.add_argument("--user-signals-from",
                         help="Borrow user signals from an existing dataset's results "
                              "(e.g., 'score_10k'). Uses data/results/{name}_{annotator}_{scheme}_turns.jsonl "
                              "instead of looking for user signal batch results in the current job.")

    args = parser.parse_args()

    if args.stage == "prepare":
        cmd_prepare(args)
    elif args.stage == "submit":
        cmd_submit(args)
    elif args.stage == "status":
        cmd_status(args)
    elif args.stage == "download":
        cmd_download(args)
    elif args.stage == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
