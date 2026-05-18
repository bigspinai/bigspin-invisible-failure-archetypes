"""
Use Case / Industry Tagger

Tags chat transcripts with industry vertical and use case category.
Designed to pair with UX signal tags for industry-level quality analysis.

Usage:
    python tag_use_cases.py submit  input.csv --job-name verticals
    python tag_use_cases.py status  verticals
    python tag_use_cases.py results verticals -o use_cases.csv
    python tag_use_cases.py run     input.csv --sample 20

Note: This tagger works on the transcript_summary field if available
(from UX signal output), falling back to the full transcript.
This is cheaper and faster since summaries are ~1 sentence.

Requires:
    pip install anthropic openai
    Set ANTHROPIC_API_KEY or OPENAI_API_KEY env var.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# Added by Chris to handle some looooooong transcripts:
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data analyst classifying chat transcripts by industry vertical and use case.

## Your Task

Read the provided chat transcript (or its summary) and assign it to an industry vertical and use case category.

## Industry Verticals

Assign exactly ONE primary vertical. If the conversation clearly spans two, assign a secondary as well. Choose from:

### Technology & Engineering
**`software_development`** — Writing, debugging, reviewing, or architecting code. Includes web/mobile/game development, DevOps, databases, APIs, system design, and programming language questions.

**`data_science_ml`** — Data analysis, machine learning, statistics, model training, data pipelines, visualization, Jupyter notebooks, pandas/numpy work, AI/ML concepts.

**`it_infrastructure`** — System administration, networking, cloud services, server configuration, security, hardware troubleshooting, operating systems.

### Professional Services
**`legal`** — Contracts, compliance, regulatory questions, intellectual property, legal document drafting, case analysis, jurisdiction questions.

**`finance_accounting`** — Investment analysis, financial modeling, accounting, budgeting, trading, tax questions, economic analysis, banking.

**`consulting_strategy`** — Business strategy, management consulting, competitive analysis, market sizing, frameworks (SWOT, Porter's, etc.).

### Business Operations
**`marketing_advertising`** — Marketing campaigns, ad copy, SEO, social media content, brand strategy, market research, content calendars.

**`sales_crm`** — Sales outreach, CRM usage, lead generation, pipeline management, proposal writing, cold emails.

**`hr_recruiting`** — Job descriptions, interview prep, resume/cover letter writing, employee policies, onboarding, performance reviews.

**`customer_support`** — Handling customer complaints, writing support responses, ticket triage, FAQ creation, escalation procedures.

**`product_management`** — Product specs, roadmaps, user stories, feature prioritization, A/B testing, product strategy.

### Domain Verticals
**`healthcare_medical`** — Medical questions, health advice, pharmaceutical info, clinical research, patient care, medical terminology, fitness/nutrition in a health context.

**`education_academic`** — Homework help, academic essays, research papers, study guides, exam prep, educational content creation, curriculum design.

**`ecommerce_retail`** — Product listings, inventory management, pricing strategy, marketplace optimization, shopping recommendations.

**`real_estate`** — Property listings, market analysis, mortgage questions, investment properties, real estate transactions.

**`manufacturing_supply_chain`** — Production processes, logistics, inventory optimization, quality control, supply chain management.

**`energy_utilities`** — Energy production, sustainability, utility management, renewable energy, grid operations.

**`government_public_sector`** — Policy analysis, public administration, government processes, civic engagement, regulatory compliance.

**`travel_hospitality`** — Travel planning, hotel/restaurant operations, tourism, booking systems, travel guides.

**`insurance`** — Policy questions, claims processing, risk assessment, underwriting, insurance product comparison.

**`telecommunications`** — Network operations, service provisioning, telecom infrastructure, mobile services.

### Content & Creative
**`creative_writing`** — Fiction, poetry, screenplays, storytelling, worldbuilding, fan fiction, narrative craft. The primary intent is creating a creative work.

**`content_production`** — Blog posts, articles, copywriting, editing, technical writing, documentation. Professional content creation (not academic).

**`media_entertainment`** — Film/TV/music discussion, entertainment industry, media production, celebrity/pop culture analysis.

**`design_ux`** — UI/UX design, graphic design, design tools, user research, wireframing, design systems.

### Personal & General
**`personal_lifestyle`** — Personal advice, hobbies, recipes, relationships, travel planning (personal), general life questions.

**`gaming`** — Video games, game mechanics, esports, game walkthroughs, character builds, gaming culture.

**`general_knowledge`** — Trivia, factual questions, history, geography, science explainers, curiosity-driven questions without a professional context.

**`translation_language`** — Translation between languages, language learning, grammar questions, localization.

**`math_science`** — Mathematics, physics, chemistry, biology questions in an academic or curiosity context (not tied to a specific industry).

### Special
**`adversarial_invalid`** — Prompt injection, jailbreak attempts, DAN prompts, deliberate boundary-testing, or content that is clearly not a legitimate use of the AI.

**`explicit_nsfw`** — Sexually explicit content, fetish content, or content designed primarily to generate graphic material.

**`ambiguous_unclear`** — Cannot determine the use case from the available information. Garbled input, extremely vague, or truly unclassifiable.

## Output Format

Return a JSON object:

```json
{
  "primary_vertical": "software_development",
  "secondary_vertical": null,
  "professional_context": "yes",
  "use_case_brief": "Debugging a Python web scraping script that throws timeout errors",
  "confidence": "high"
}
```

**Field definitions:**
- `primary_vertical`: The single best-fitting vertical from the list above.
- `secondary_vertical`: A second vertical ONLY if the conversation clearly spans two (e.g., a marketing question about healthcare). null otherwise.
- `professional_context`: "yes" if this looks like work/professional use, "no" if personal/hobby, "ambiguous" if unclear. A student doing homework is "no". Someone writing a professional report is "yes".
- `use_case_brief`: One sentence describing what the user is specifically doing (not just the category — be specific).
- `confidence`: "high" if the classification is clear, "medium" if reasonable but could go another way, "low" if it's genuinely ambiguous.

## Guidelines

1. **Classify by user intent, not topic.** A student asking about medicine for homework is `education_academic`, not `healthcare_medical`. A professional writing medical documentation is `healthcare_medical`.
2. **When in doubt about professional context, lean toward the topic vertical.** A coding question is `software_development` whether it's for work or a side project.
3. **Creative writing requests go to `creative_writing`** unless the content is clearly professional content production (e.g., "write a blog post for our company").
4. **Multi-turn conversations:** classify based on the dominant use case, not individual turns.
5. **Output valid JSON only.** No commentary, no markdown, no explanation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

OUTPUT_FIELDS = [
    "conversation_id",
    "model",
    "primary_vertical",
    "secondary_vertical",
    "professional_context",
    "use_case_brief",
    "confidence",
    "full_response_json",
    "input_text",
]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _load_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"Error: {p} not found.", file=sys.stderr)
        sys.exit(1)
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get_input_text(row: dict) -> str:
    """Use transcript_summary if available (cheaper), else full transcript."""
    if row.get("transcript_summary"):
        return row["transcript_summary"]
    if row.get("transcript"):
        return str(row["transcript"])
    # Fallback: use whatever text fields exist
    return str(row)


def _get_conversation_id(row: dict) -> str:
    return str(row.get("conversation_id", row.get("id", "")))


def _sample_rows(rows, n, seed):
    if n is None:
        return rows
    n = min(n, len(rows))
    random.seed(seed)
    return random.sample(rows, n)


def _result_row(cid, input_text, model, tags):
    return {
        "conversation_id": cid,
        "model": model,
        "primary_vertical": tags.get("primary_vertical", ""),
        "secondary_vertical": tags.get("secondary_vertical", ""),
        "professional_context": tags.get("professional_context", ""),
        "use_case_brief": tags.get("use_case_brief", ""),
        "confidence": tags.get("confidence", ""),
        "full_response_json": json.dumps(tags),
        "input_text": input_text,
    }


def _write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(results):
    verticals = [r["primary_vertical"] for r in results if r["primary_vertical"]]
    if not verticals:
        return
    from collections import Counter
    counts = Counter(verticals)
    print("\nVertical distribution:")
    for v, c in counts.most_common():
        pct = c / len(verticals) * 100
        bar = "█" * int(pct / 2)
        print(f"  {v:35s} {c:5d} ({pct:5.1f}%) {bar}")

    pro = [r["professional_context"] for r in results if r["professional_context"]]
    if pro:
        pro_counts = Counter(pro)
        print(f"\nProfessional context: {pro_counts.get('yes',0)} yes, {pro_counts.get('no',0)} no, {pro_counts.get('ambiguous',0)} ambiguous")


# ---------------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------------

def cmd_submit(args):
    from anthropic import Anthropic

    rows = _load_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows from {args.input_csv}")

    sample = _sample_rows(rows, args.sample, args.seed)

    if args.exclude_from:
        exclude_path = Path(args.exclude_from)
        if not exclude_path.exists():
            print(f"Error: {exclude_path} not found.", file=sys.stderr)
            sys.exit(1)
        with open(exclude_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            done_ids = {
                r["conversation_id"]
                for r in reader
                if r.get("primary_vertical")
            }
        before = len(sample)
        sample = [r for r in sample if _get_conversation_id(r) not in done_ids]
        print(f"Excluded {before - len(sample)} already-tagged (from {exclude_path})")
        print(f"Remaining: {len(sample)}")
        if not sample:
            print("Nothing to submit — all transcripts already tagged!")
            return

    model = args.model or DEFAULT_MODEL
    chunk_size = args.chunk_size

    print(f"Total: {len(sample)} | Chunk size: {chunk_size}")
    num_chunks = (len(sample) + chunk_size - 1) // chunk_size
    print(f"Will submit {num_chunks} batch(es) with {model}...")

    all_requests = []
    for row in sample:
        all_requests.append({
            "custom_id": _get_conversation_id(row),
            "params": {
                "model": model,
                "max_tokens": 512,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": _get_input_text(row)}],
            },
        })

    client = Anthropic()
    job_name = args.job_name or f"usecase_{int(time.time())}"
    meta_path = Path(f"job_{job_name}.json")

    if meta_path.exists() and not args.fresh:
        with open(meta_path) as f:
            meta = json.load(f)
        batch_ids = meta.get("batch_ids", [])
        start_chunk = len(batch_ids)
        print(f"\n  Resuming '{job_name}' — {start_chunk} chunk(s) already submitted.")
    else:
        batch_ids = []
        start_chunk = 0

    def _save_meta():
        meta = {
            "job_name": job_name,
            "model": model,
            "num_requests": len(all_requests),
            "num_batches": len(batch_ids),
            "batch_ids": batch_ids,
            "input_csv": args.input_csv,
            "chunk_size": chunk_size,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tagger": "use_case",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    max_retries = 3
    for i in range(start_chunk * chunk_size, len(all_requests), chunk_size):
        chunk = all_requests[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"\n  Chunk {chunk_num}/{num_chunks} ({len(chunk)} requests)...", end=" ", flush=True)

        for attempt in range(1, max_retries + 1):
            try:
                batch = client.messages.batches.create(requests=chunk)
                batch_ids.append(batch.id)
                print(f"✓ {batch.id}")
                _save_meta()
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * attempt
                    print(f"\n    ✗ Attempt {attempt}: {e}\n    Retrying in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"\n    ✗ All {max_retries} attempts failed: {e}")
                    _save_meta()
                    print(f"\n  Partial job saved. Resume with: python tag_use_cases.py submit {args.input_csv} --job-name {job_name}")
                    sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Job: {job_name} | {len(batch_ids)} batches | {len(all_requests)} requests")
    print(f"{'='*60}")
    print(f"\n  Status:  python tag_use_cases.py status  {job_name}")
    print(f"  Results: python tag_use_cases.py results {job_name} -o use_cases.csv")


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------

def _load_job_meta(job_id):
    meta_path = Path(f"job_{job_id}.json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"batch_ids": [job_id], "job_name": job_id}


def cmd_status(args):
    from anthropic import Anthropic

    meta = _load_job_meta(args.job_id)
    batch_ids = meta["batch_ids"]
    client = Anthropic()

    total_all = done_all = succeeded_all = errored_all = 0
    all_ended = True

    print(f"Job: {meta.get('job_name', args.job_id)} ({len(batch_ids)} batch(es))\n")
    for bid in batch_ids:
        batch = client.messages.batches.retrieve(bid)
        rc = batch.request_counts
        total = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
        done = rc.succeeded + rc.errored
        pct = (done / total * 100) if total > 0 else 0
        icon = "✓" if batch.processing_status == "ended" else "⏳"
        print(f"  {icon} {bid}  {done:>6}/{total:<6} ({pct:5.1f}%)  {batch.processing_status}")
        total_all += total
        done_all += done
        succeeded_all += rc.succeeded
        errored_all += rc.errored
        if batch.processing_status != "ended":
            all_ended = False

    pct_all = (done_all / total_all * 100) if total_all > 0 else 0
    print(f"\nOverall: {done_all}/{total_all} ({pct_all:.1f}%) — {succeeded_all} succeeded, {errored_all} errored")

    if all_ended:
        print(f"\nDone! python tag_use_cases.py results {args.job_id} -o use_cases.csv")

    if args.wait and not all_ended:
        print(f"\nPolling every {args.poll}s...")
        while not all_ended:
            time.sleep(args.poll)
            done_all = 0
            all_ended = True
            for bid in batch_ids:
                batch = client.messages.batches.retrieve(bid)
                rc = batch.request_counts
                done_all += rc.succeeded + rc.errored
                if batch.processing_status != "ended":
                    all_ended = False
            print(f"  [{time.strftime('%H:%M:%S')}] {done_all}/{total_all} ({done_all/total_all*100:.1f}%)")
        print("All batches complete!")


# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

def cmd_results(args):
    from anthropic import Anthropic

    meta = _load_job_meta(args.job_id)
    batch_ids = meta["batch_ids"]
    model = meta.get("model", "unknown")
    client = Anthropic()

    not_done = [bid for bid in batch_ids if client.messages.batches.retrieve(bid).processing_status != "ended"]
    if not_done:
        if not args.wait:
            print(f"{len(not_done)}/{len(batch_ids)} still processing. Use --wait or check later.")
            sys.exit(1)
        print(f"Waiting (every {args.poll}s)...")
        while not_done:
            time.sleep(args.poll)
            not_done = [bid for bid in not_done if client.messages.batches.retrieve(bid).processing_status != "ended"]
            print(f"  [{time.strftime('%H:%M:%S')}] {len(batch_ids)-len(not_done)}/{len(batch_ids)} complete")

    # Load input texts
    input_texts = {}
    if meta.get("input_csv"):
        try:
            for row in _load_csv(meta["input_csv"]):
                cid = _get_conversation_id(row)
                input_texts[cid] = _get_input_text(row)
        except Exception:
            pass

    results = []
    errors = 0
    for batch_num, bid in enumerate(batch_ids, 1):
        print(f"Downloading {batch_num}/{len(batch_ids)} ({bid})...", end=" ", flush=True)
        count = 0
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            input_text = input_texts.get(cid, "")
            if result.result.type == "succeeded":
                try:
                    tags = _parse_json(result.result.message.content[0].text)
                except Exception:
                    tags = {"error": "parse_error"}
                    errors += 1
            else:
                tags = {"error": result.result.type}
                errors += 1
            results.append(_result_row(cid, input_text, model, tags))
            count += 1
        print(f"✓ {count} rows")

    _write_csv(results, Path(args.output))
    print(f"\n{len(results)} results ({errors} errors) → {args.output}")
    _print_summary(results)


# ---------------------------------------------------------------------------
# RUN (synchronous)
# ---------------------------------------------------------------------------

def cmd_run(args):
    rows = _load_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows")

    sample = _sample_rows(rows, args.sample, args.seed)
    model = args.model

    if args.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        model = model or DEFAULT_MODEL

        def call(text):
            resp = client.messages.create(
                model=model, max_tokens=512, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            return _parse_json(resp.content[0].text)
    elif args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = model or "gpt-4o-mini"

        def call(text):
            resp = client.chat.completions.create(
                model=model, max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            return _parse_json(resp.choices[0].message.content)

    print(f"Tagging {len(sample)} with {model}...\n")
    results = []
    for i, row in enumerate(sample):
        cid = _get_conversation_id(row)
        input_text = _get_input_text(row)
        print(f"[{i+1}/{len(sample)}] {cid[:30]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            tags = call(input_text)
            print(f"✓ {tags.get('primary_vertical','')} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"✗ {e} ({time.time()-t0:.1f}s)")
            tags = {"error": str(e)}
        results.append(_result_row(cid, input_text, model, tags))

    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_verticals.csv")
    _write_csv(results, output_path)
    print(f"\nDone → {output_path}")
    _print_summary(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tag transcripts by industry vertical and use case.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="Submit batch job")
    p.add_argument("input_csv")
    p.add_argument("--model", default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--job-name", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--exclude-from", default=None)

    p = sub.add_parser("status", help="Check progress")
    p.add_argument("job_id")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll", type=int, default=30)

    p = sub.add_parser("results", help="Download results")
    p.add_argument("job_id")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll", type=int, default=30)

    p = sub.add_parser("run", help="Tag synchronously (testing)")
    p.add_argument("input_csv")
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default=None)
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", default=None)

    args = parser.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "results": cmd_results, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
