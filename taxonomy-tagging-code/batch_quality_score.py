#!/usr/bin/env python3
"""
Batch quality-scoring pipeline for transcript-level classification.

Takes turn-tagged transcript files (from data/results/) and submits batch
jobs that classify each transcript for overall quality + failure archetypes.

Supports two conditions:
  B — Transcript text + inline turn-level signals + signal summary
  C — Signal profile only (no transcript text)

Input signals are majority-voted across annotators (≥2/3 agreement).

Pipeline stages (same lifecycle as batch_score.py):
    prepare   — Build batch request files from tagged transcripts
    submit    — Submit batch files to Anthropic / OpenAI APIs
    status    — Check status of submitted batches
    download  — Download completed batch results
    merge     — Parse results into unified quality-scored output

Usage:
    # Prepare condition B+C for score_10k, all taggers
    python batch_quality_score.py prepare --dataset score_10k

    # Prepare only condition B, opus tagger
    python batch_quality_score.py prepare --dataset score_10k --condition B --tagger opus

    # Dry run
    python batch_quality_score.py prepare --dataset score_10k --dry-run

    # Submit / status / download / merge
    python batch_quality_score.py submit   --dataset score_10k
    python batch_quality_score.py status   --dataset score_10k
    python batch_quality_score.py download --dataset score_10k
    python batch_quality_score.py merge    --dataset score_10k
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Configs ───────────────────────────────────────────────────────────

# Taggers used for quality classification (same models as turn-tagging)
TAGGERS = {
    "opus":   {"provider": "anthropic", "model": "claude-opus-4-6",   "label": "Opus 4.6"},
    "sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-6", "label": "Sonnet 4.6"},
    "gpt5":   {"provider": "openai",   "model": "gpt-5.4",           "label": "GPT-5.4"},
}

CONDITIONS = ["A", "B", "C"]

BATCH_SIZE = 2000  # requests per batch file

# Which annotators provided turn-level signals for each dataset
DATASET_ANNOTATORS = {
    "score_10k": ["opus", "gpt5", "sonnet"],
    "score_30k": ["opus", "gpt5"],
    "score_59k": ["opus", "gpt5"],
    "test_1k":   ["opus", "gpt5", "sonnet"],
    "wildchat4M-temporal-score": ["opus", "gpt5", "sonnet"]
}

# Default scheme to use for signal inputs
DEFAULT_SCHEME = "calibrated"

# ── Signal valence (from stage2_experiment/run_stage2.py) ─────────────

POSITIVE_SIGNALS = {
    'conversation_advanced', 'user_empowered', 'adaptation', 'error_recovery',
    'appropriate_hedge', 'appropriate_confidence', 'ai_asked_clarifying_question',
    'ai_acknowledges_correction',
}
NEGATIVE_SIGNALS = {
    'under_delivered', 'false_confidence', 'conversation_stalled', 'repetition',
    'error_commitment', 'user_misled', 'intent_missed', 'off_topic_drift',
    'silent_assumption', 'ai_self_contradiction', 'ai_implicit_refusal', 'ai_malfunction',
    'plow_through', 'performative_hedge', 'over_delivered', 'problem_ignored',
    'generate_without_clarifying', 'factual_error',
}
USER_REACTION_SIGNALS = {
    'user_corrects_ai', 'user_expresses_frustration',
    'user_expresses_dissatisfaction', 'user_asks_clarification',
}

# ── System prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert analyst of human-AI conversations. Your task is to classify a conversation transcript on three dimensions:

1. OVERALL QUALITY — how well did the AI serve the user?
2. FAILURE VISIBILITY — if something went wrong, did the user notice?
3. FAILURE ARCHETYPES — which failure patterns, if any, are present?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIMENSION 1: OVERALL QUALITY (choose exactly one)

good — The user's task was accomplished with no significant issues. The AI understood the goal, provided accurate and relevant content, and the user could act on it. Minor imperfections that don't affect the outcome are acceptable.

acceptable — The task was accomplished but with manageable problems. The user got value, but there were notable issues: excessive verbosity, minor inaccuracies, partially missed requirements, or unnecessary detours. The user had to work around the AI's limitations but ultimately could achieve their goal.

poor — The task was not accomplished or was significantly flawed. The AI misunderstood the goal, provided substantially incorrect information, failed to address key requirements, or the conversation broke down. The user would need to start over or seek help elsewhere for the core task.

critical — The task failed AND the AI's output could actively mislead. Wrong answers delivered with confidence, professional-looking output that misses the point entirely, or the user left the conversation holding incorrect information they're likely to act on. The distinguishing factor is confident incorrectness that the user is unlikely to catch.

KEY BOUNDARY — acceptable vs poor: Did the user get enough value to accomplish their core goal, even imperfectly? If yes → acceptable. If no → poor. The question is not whether the response was good, but whether it was usable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIMENSION 2: FAILURE VISIBILITY (choose exactly one)

This dimension asks: if something went wrong in this conversation, did the user notice?

none — No failure occurred. The conversation succeeded (typically aligns with quality=good, but use your judgment).

visible — The user noticed and reacted to the failure. Look for: explicit corrections, expressions of frustration or dissatisfaction, pointed requests for clarification, the user restating their original request, or direct statements that the AI got something wrong. The user's behavior changed in response to the problem.

invisible — A failure occurred but the user did not catch it or react to it. The conversation may look successful on the surface — the user seems satisfied, doesn't push back, and may even thank the AI — but the AI's output was wrong, off-target, or inadequate in ways the user didn't flag. This is the most dangerous category: the user walks away with a false sense of success.

mixed — The conversation contains BOTH visible and invisible failures. The user caught and reacted to some problems but missed others. For example: the user corrected a factual error (visible) but didn't notice the AI silently dropped one of their requirements (invisible).

IMPORTANT: Failure visibility is independent of quality. A "poor" conversation can have invisible failures (user didn't realize how bad it was) or visible failures (user noticed and complained). An "acceptable" conversation can have visible failures (user caught and worked around issues).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIMENSION 3: FAILURE ARCHETYPES (select all that apply, or "none")

These describe WHAT went wrong — the structural pattern of the failure.

The Confidence Trap — The AI presents incorrect information with unwarranted certainty, and the user accepts it without challenge. The danger is that the interaction looks successful — the AI sounds authoritative, the user seems satisfied — but the user walks away with wrong information. Look for: factual errors delivered without hedging, fabricated specifics, user building on incorrect premises.

The Silent Mismatch — The AI addresses a different goal than the user intended, but the response is plausible enough that neither party flags the disconnect. The AI "answers a question the user didn't ask." Look for: response that is competent but off-target, user's actual need going unaddressed, subtle misinterpretation of the request.

The Drift — The conversation gradually or abruptly loses its connection to the user's original goal. The AI may elaborate on tangentially related topics, add unrequested content, or respond to its own interpretation rather than the user's intent. Can be gradual (verbosity creeping off-topic over turns) or sudden (AI addresses a different but related topic). Look for: response relevance declining over turns, AI addressing adjacent but wrong goals, user's specific requirements getting lost.

The Death Spiral — The conversation enters a repetitive loop. The user keeps asking or correcting, and the AI keeps producing the same or similar output without incorporating the feedback. No progress despite continued effort. Look for: repeated similar responses across turns, user corrections that don't result in changes, escalating user frustration with static AI behavior.

The Contradiction Unravel — The AI contradicts its own prior statements without acknowledging the change. In earlier turns it said X; now it says not-X, with no "I was wrong" or "on reflection." This can erode the user's ability to determine what's actually correct. Look for: incompatible claims across turns, unstated reversals, user potentially confused about which version to trust.

The Walkaway — The conversation ends without resolution and without the user explicitly signaling failure. The user simply stops engaging. This is the hardest archetype to identify because the absence of a signal IS the signal. Look for: unresolved user goal, final AI response that doesn't fully address the need, no subsequent user message, conversation ending without natural closure.

The Partial Recovery — The conversation hits a clear failure but partially recovers. The AI or user identifies the problem and course-corrects, but the recovery is incomplete — the user gets some value but the original goal is not fully met. Look for: error followed by correction, improvement in response quality across turns, but remaining gaps or unaddressed aspects.

The Mystery Failure — The user's goal was not achieved, but no specific failure pattern from the above list explains why. The conversation just... didn't work, and it's hard to point to a specific breakdown. This is a catch-all that flags gaps in our analytical framework. Use sparingly — prefer a specific archetype if one fits even partially.

None — No failure pattern detected. The conversation succeeded.

CLASSIFICATION RULES:
- Choose exactly ONE quality category
- Choose exactly ONE failure visibility category
- Choose ONE OR MORE archetypes (including "none" if appropriate)
- All three dimensions are independent: classify each on its own merits
- Multiple archetypes can co-occur
- For each archetype you select, cite specific turns as evidence
- For quality and failure visibility, explain your reasoning in 1-2 sentences"""


JSON_INSTRUCTION = """
Respond in this exact JSON format:
{
  "quality": "good|acceptable|poor|critical",
  "quality_rationale": "1-2 sentence explanation",
  "failure_visibility": "none|visible|invisible|mixed",
  "failure_visibility_rationale": "1-2 sentence explanation",
  "archetypes": [
    {
      "archetype": "archetype_name",
      "evidence_turns": [turn numbers],
      "rationale": "1-2 sentence explanation citing specific turns"
    }
  ],
  "archetype_count": <integer>
}

If no failure archetypes apply, use:
"archetypes": [{"archetype": "none", "evidence_turns": [], "rationale": "Conversation succeeded — user's goal was met without significant issues."}]

Respond with ONLY the JSON object."""


# ── Data loading ──────────────────────────────────────────────────────

def load_transcript_texts(dataset_name: str) -> dict:
    """Load raw transcript text from the original dataset.

    Returns: {conversation_id: [sorted list of turn dicts with user_content, ai_content]}
    """
    dataset_path = SCRIPT_DIR / "data" / "datasets" / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        sys.exit(1)

    by_conv = defaultdict(list)
    with open(dataset_path) as f:
        for line in f:
            turn = json.loads(line)
            by_conv[turn["conversation_id"]].append(turn)

    # Sort turns within each conversation
    for cid in by_conv:
        by_conv[cid].sort(key=lambda t: t["turn_number"])

    return dict(by_conv)


def load_annotator_signals(dataset_name: str, scheme: str = DEFAULT_SCHEME) -> dict:
    """Load turn-level signals from all available annotators for a dataset.

    Returns: {conversation_id: {turn_number: {annotator: {"ai": [...], "user": [...]}}}}
    """
    results_dir = SCRIPT_DIR / "data" / "results"
    annotator_keys = DATASET_ANNOTATORS.get(dataset_name, [])

    #print("annotator_keys", annotator_keys)

    signals = defaultdict(lambda: defaultdict(dict))

    for ann in annotator_keys:
        transcript_file = results_dir / f"{dataset_name}_{ann}_{scheme}_transcripts.jsonl"
        if not transcript_file.exists():
            print(f"  WARNING: Missing {transcript_file.name}")
            continue

        with open(transcript_file) as f:
            for line in f:
                rec = json.loads(line)
                cid = rec["conversation_id"]
                for turn in rec["turns"]:
                    tn = turn["turn_number"]
                    ai_sigs = [s["signal"] for s in turn.get("ai_signals", [])]
                    user_sigs = [s["signal"] for s in turn.get("user_signals", [])]
                    signals[cid][tn][ann] = {"ai": ai_sigs, "user": user_sigs}

    return dict(signals)


def majority_vote_signals(turn_signals_by_annotator: dict) -> tuple[list, list]:
    """Majority-vote across annotators for a single turn.

    Returns: (ai_majority_vote_list, user_majority_vote_list)
    Each list contains (signal_name, vote_count) tuples where votes >= 2.
    """
    ai_votes = Counter()
    user_votes = Counter()

    for ann, sigs in turn_signals_by_annotator.items():
        for s in sigs.get("ai", []):
            ai_votes[s] += 1
        for s in sigs.get("user", []):
            user_votes[s] += 1

    n_annotators = len(turn_signals_by_annotator)
    threshold = 2 if n_annotators >= 3 else 1  # require majority; fallback for 2-annotator case

    ai_mv = [(s, v) for s, v in ai_votes.items() if v >= threshold]
    user_mv = [(s, v) for s, v in user_votes.items() if v >= threshold]

    return ai_mv, user_mv


# ── Prompt builders ───────────────────────────────────────────────────

def _format_signal_marker(sig: str, votes: int, total: int = 3) -> str:
    if votes == total:
        return f"{sig} ✓"
    return f"{sig} ⚠({votes}/{total})"


def _build_signal_summary(turn_texts: list, signals: dict, n_annotators: int) -> str:
    """Build the SIGNAL SUMMARY section used by conditions B and C."""
    sig_turn_map = defaultdict(list)

    for t in turn_texts:
        tn = t["turn_number"]
        turn_sigs = signals.get(tn, {})
        ai_mv, user_mv = majority_vote_signals(turn_sigs)
        for sig, votes in ai_mv + user_mv:
            sig_turn_map[sig].append((tn, votes))

    pos_sigs = {s: v for s, v in sig_turn_map.items() if s in POSITIVE_SIGNALS}
    neg_sigs = {s: v for s, v in sig_turn_map.items() if s in NEGATIVE_SIGNALS}

    lines = ["SIGNAL SUMMARY ACROSS TRANSCRIPT:", ""]

    # Positive
    lines.append("Positive signals:")
    if pos_sigs:
        for sig in sorted(pos_sigs):
            entries = pos_sigs[sig]
            parts = [f"{tn} ({v}/{n_annotators})" for tn, v in entries]
            lines.append(f"  {sig}: turns {', '.join(parts)}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Negative
    lines.append("Negative signals:")
    if neg_sigs:
        for sig in sorted(neg_sigs):
            entries = neg_sigs[sig]
            parts = [f"{tn} ({v}/{n_annotators})" for tn, v in entries]
            lines.append(f"  {sig}: turns {', '.join(parts)}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Trajectory
    lines.append("Trajectory:")
    ca_turns = [tn for tn, v in sig_turn_map.get("conversation_advanced", [])]
    lines.append(f"  - conversation_advanced present on: "
                 f"{', '.join(map(str, ca_turns)) if ca_turns else 'no turns'}")

    first_neg = None
    for t in turn_texts:
        tn = t["turn_number"]
        turn_sigs = signals.get(tn, {})
        ai_mv, user_mv = majority_vote_signals(turn_sigs)
        for sig, votes in ai_mv + user_mv:
            if sig in NEGATIVE_SIGNALS:
                first_neg = (tn, sig)
                break
        if first_neg:
            break
    lines.append(f"  - First negative signal: "
                 f"{'turn ' + str(first_neg[0]) + ' (' + first_neg[1] + ')' if first_neg else 'none'}")

    reactions = []
    for t in turn_texts:
        tn = t["turn_number"]
        turn_sigs = signals.get(tn, {})
        _, user_mv = majority_vote_signals(turn_sigs)
        for sig, votes in user_mv:
            if sig in USER_REACTION_SIGNALS:
                reactions.append(f"{sig} (turn {tn})")
    lines.append(f"  - User reaction signals: {', '.join(reactions) if reactions else 'none'}")
    lines.append("  - Final turn status: no subsequent user message")
    lines.append("")

    # Aggregate
    lines.append("Aggregate:")
    n_turns = len(turn_texts)
    pos_turn_count = 0
    neg_turn_count = 0
    for t in turn_texts:
        tn = t["turn_number"]
        turn_sigs = signals.get(tn, {})
        ai_mv, user_mv = majority_vote_signals(turn_sigs)
        all_mv = ai_mv + user_mv
        has_pos = any(s in POSITIVE_SIGNALS for s, _ in all_mv)
        has_neg = any(s in NEGATIVE_SIGNALS for s, _ in all_mv)
        if has_pos:
            pos_turn_count += 1
        if has_neg:
            neg_turn_count += 1

    lines.append(f"  - Total turns: {n_turns}")
    lines.append(f"  - Turns with any positive signal: {pos_turn_count}/{n_turns}")
    lines.append(f"  - Turns with any negative signal: {neg_turn_count}/{n_turns}")
    lines.append(f"  - Unique negative signals: {len(neg_sigs)}")

    return "\n".join(lines)


def build_condition_a(turn_texts: list, signals: dict, n_annotators: int) -> str:
    """Condition A: transcript only (no signals)."""
    lines = [
        "Classify the following conversation transcript. Use the conversation "
        "text alone to inform your classification.",
        "",
        "TRANSCRIPT:",
    ]

    for t in turn_texts:
        tn = t["turn_number"]
        lines.append(f"<turn n=\"{tn}\">")
        lines.append(f"[User]: {t['user_content']}")
        lines.append(f"[AI]: {t['ai_content']}")
        lines.append("</turn>")
        lines.append("")

    lines.append("[Note: No subsequent user message after the final AI response.]")
    lines.append("")
    lines.append(JSON_INSTRUCTION)
    return "\n".join(lines)


def build_condition_b(turn_texts: list, signals: dict, n_annotators: int) -> str:
    """Condition B: transcript + inline signals + signal summary."""
    lines = [
        "Classify the following conversation transcript. You are also provided with "
        "behavioral signals extracted from each turn by an independent annotation system. "
        "Use both the transcript text and the extracted signals to inform your classification.",
        "",
        "SIGNAL LEGEND:",
        f"  ✓ = all {n_annotators} independent annotators agreed this signal is present (high confidence)",
        f"  ⚠ = {n_annotators - 1} of {n_annotators} annotators agreed (moderate confidence — use judgment)",
        "  Signals not listed for a turn were agreed to be absent by all annotators.",
        "",
        "TRANSCRIPT WITH EXTRACTED SIGNALS:",
    ]

    for t in turn_texts:
        tn = t["turn_number"]
        lines.append(f"<turn n=\"{tn}\">")
        lines.append(f"[User]: {t['user_content']}")
        lines.append(f"[AI]: {t['ai_content']}")

        turn_sigs = signals.get(tn, {})
        ai_mv, user_mv = majority_vote_signals(turn_sigs)
        all_mv = ai_mv + user_mv

        if all_mv:
            parts = [_format_signal_marker(s, v, n_annotators)
                     for s, v in sorted(all_mv, key=lambda x: x[0])]
            lines.append(f"<signals>{', '.join(parts)}</signals>")
        else:
            lines.append("<signals>none detected</signals>")
        lines.append("</turn>")
        lines.append("")

    lines.append("[Note: No subsequent user message after the final AI response.]")
    lines.append("")
    lines.append(_build_signal_summary(turn_texts, signals, n_annotators))
    lines.append("")
    lines.append(JSON_INSTRUCTION)
    return "\n".join(lines)


def build_condition_c(turn_texts: list, signals: dict, n_annotators: int) -> str:
    """Condition C: signals only (no transcript text)."""
    lines = [
        "Classify the following conversation based on its behavioral signal profile. "
        "You are provided with turn-by-turn behavioral signals extracted by an independent "
        "annotation system, but NOT the raw conversation text. Use the signal patterns, "
        "trajectory, and co-occurrences to classify the conversation.",
        "",
        "SIGNAL LEGEND:",
        f"  ✓ = all {n_annotators} independent annotators agreed this signal is present (high confidence)",
        f"  ⚠ = {n_annotators - 1} of {n_annotators} annotators agreed (moderate confidence — use judgment)",
        "  Signals not listed for a turn were agreed to be absent by all annotators.",
        "",
        "TURN-BY-TURN SIGNAL PROFILE:",
    ]

    for t in turn_texts:
        tn = t["turn_number"]
        lines.append(f"<turn n=\"{tn}\">")

        turn_sigs = signals.get(tn, {})
        ai_mv, user_mv = majority_vote_signals(turn_sigs)

        if user_mv:
            parts = [_format_signal_marker(s, v, n_annotators)
                     for s, v in sorted(user_mv, key=lambda x: x[0])]
            lines.append(f"  User turn signals: {', '.join(parts)}")
        else:
            lines.append("  User turn signals: (none detected)")

        if ai_mv:
            parts = [_format_signal_marker(s, v, n_annotators)
                     for s, v in sorted(ai_mv, key=lambda x: x[0])]
            lines.append(f"  AI turn signals: {', '.join(parts)}")
        else:
            lines.append("  AI turn signals: (none detected)")
        lines.append("</turn>")
        lines.append("")

    lines.append("[Note: No subsequent user message after the final AI response.]")
    lines.append("")
    lines.append(_build_signal_summary(turn_texts, signals, n_annotators))
    lines.append("")
    lines.append(JSON_INSTRUCTION)
    return "\n".join(lines)


PROMPT_BUILDERS = {
    "A": build_condition_a,
    "B": build_condition_b,
    "C": build_condition_c,
}


# ── Batch file formatting (provider-specific) ────────────────────────

def format_anthropic_request(custom_id: str, prompt: str, system: str, model: str,
                              max_tokens: int = 2048) -> dict:
    """Format a request for Anthropic's Message Batches API."""
    messages = [{"role": "user", "content": prompt}]
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": messages,
    }
    if system:
        params["system"] = system
    return {"custom_id": custom_id, "params": params}


def format_openai_request(custom_id: str, prompt: str, system: str, model: str,
                           max_tokens: int = 2048) -> dict:
    """Format a request for OpenAI's Batch API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "messages": messages,
        },
    }


# ── Job directory structure ───────────────────────────────────────────

def get_job_dir(dataset_name: str) -> Path:
    """Get/create the quality-scoring batch job directory."""
    job_dir = SCRIPT_DIR / "data" / "batch_jobs_quality" / dataset_name
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


# ── Stage: prepare ────────────────────────────────────────────────────

def cmd_prepare(args):
    """Build batch request JSONL files for quality classification."""
    dataset_name = args.dataset
    conditions = [args.condition] if args.condition else CONDITIONS
    tagger_keys = [args.tagger] if args.tagger else list(TAGGERS.keys())
    scheme = args.scheme

    print("=" * 70)
    print(f"PREPARE QUALITY SCORING BATCHES")
    print(f"  Dataset: {dataset_name}")
    print(f"  Conditions: {conditions}")
    print(f"  Taggers: {', '.join(tagger_keys)}")
    print(f"  Signal scheme: {scheme}")
    print("=" * 70)

    # Load transcript texts (needed for conditions A and B)
    need_texts = "A" in conditions or "B" in conditions
    if need_texts:
        print("\n  Loading transcript texts from original dataset...")
        transcript_texts = load_transcript_texts(dataset_name)
        print(f"    {len(transcript_texts)} conversations loaded")
    else:
        transcript_texts = {}

    # Load turn-level signals from all annotators (needed for B and C)
    need_signals = "B" in conditions or "C" in conditions
    if need_signals:
        print("  Loading annotator signals...")
        all_signals = load_annotator_signals(dataset_name, scheme)
        print(f"    {len(all_signals)} conversations with signal data")
    else:
        all_signals = {}

    # Determine conversation IDs to process
    id_sets = []
    if need_texts:
        id_sets.append(set(transcript_texts.keys()))
    if need_signals:
        id_sets.append(set(all_signals.keys()))
    if id_sets:
        conv_ids = sorted(set.intersection(*id_sets))
    else:
        conv_ids = []
    n_annotators = len(DATASET_ANNOTATORS.get(dataset_name, []))
    print(f"    {len(conv_ids)} conversations ready for scoring")
    print(f"    {n_annotators} signal annotators (majority-vote threshold: 2)")

    job_dir = get_job_dir(dataset_name)

    # Dry run: show example prompts
    if args.dry_run:
        example_cid = conv_ids[0]
        texts = transcript_texts.get(example_cid, [{"turn_number": 1, "user_content": "(N/A)", "ai_content": "(N/A)"}])
        sigs = all_signals.get(example_cid, {})

        for cond in conditions:
            prompt = PROMPT_BUILDERS[cond](texts, sigs, n_annotators)
            print(f"\n{'=' * 70}")
            print(f"CONDITION {cond} — conversation {example_cid} ({len(texts)} turns)")
            print(f"Prompt length: {len(prompt):,} chars")
            print(f"{'=' * 70}")
            print(f"\n[SYSTEM PROMPT]\n{SYSTEM_PROMPT[:200]}...\n")
            print(f"[USER PROMPT]\n{prompt[:3000]}...")

        total_calls = len(conv_ids) * len(conditions) * len(tagger_keys)
        print(f"\n{'=' * 70}")
        print(f"JOB SUMMARY (dry run)")
        print(f"  Conversations: {len(conv_ids)}")
        print(f"  Conditions: {conditions}")
        print(f"  Taggers: {tagger_keys}")
        print(f"  Total batch requests: {total_calls}")
        print(f"  Batches: ~{(total_calls + BATCH_SIZE - 1) // BATCH_SIZE} per tagger")
        print(f"{'=' * 70}")
        return

    # Build batch files
    for tagger_key in tagger_keys:
        tagger = TAGGERS[tagger_key]
        provider = tagger["provider"]
        formatter = format_anthropic_request if provider == "anthropic" else format_openai_request

        for cond in conditions:
            prefix = f"quality_{cond}_{tagger_key}"
            print(f"\n  Building {prefix}...")

            batch_idx = 0
            batch_count = 0
            batch_file = job_dir / f"{prefix}_batch_{batch_idx:03d}.jsonl"
            out_f = open(batch_file, "w")
            written_total = 0

            for i, cid in enumerate(conv_ids):
                texts = transcript_texts.get(cid, [])
                sigs = all_signals.get(cid, {})

                # For condition C, we still need turn_number info even without text
                if not texts and cond == "C":
                    # Build minimal turn list from signal keys
                    turn_numbers = sorted(sigs.keys())
                    texts = [{"turn_number": tn, "user_content": "", "ai_content": ""}
                             for tn in turn_numbers]

                if not texts:
                    continue

                prompt = PROMPT_BUILDERS[cond](texts, sigs, n_annotators)
                custom_id = f"quality_{cond}_{cid}"

                formatted = formatter(
                    custom_id=custom_id,
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    model=tagger["model"],
                )
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

                if (i + 1) % 2000 == 0:
                    print(f"      [{i+1}/{len(conv_ids)}] conversations processed...")

            out_f.close()
            if batch_count > 0:
                print(f"    Wrote {batch_file.name} ({batch_count} requests)")
            else:
                batch_file.unlink(missing_ok=True)
            print(f"    Total: {written_total} requests")

    # Write manifest
    manifest = {
        "dataset": dataset_name,
        "scheme": scheme,
        "conditions": conditions,
        "taggers": tagger_keys,
        "n_conversations": len(conv_ids),
        "n_signal_annotators": n_annotators,
        "signal_annotators": DATASET_ANNOTATORS.get(dataset_name, []),
        "batch_size": BATCH_SIZE,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
    }
    with open(job_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Job directory: {job_dir}")
    print(f"  Next: python batch_quality_score.py submit --dataset {dataset_name}")


# ── Stage: submit ─────────────────────────────────────────────────────

def cmd_submit(args):
    """Submit batch files to APIs."""
    dataset_name = args.dataset
    job_dir = get_job_dir(dataset_name)
    conditions = [args.condition] if args.condition else CONDITIONS
    tagger_keys = [args.tagger] if args.tagger else list(TAGGERS.keys())

    if not (job_dir / "manifest.json").exists():
        print(f"ERROR: Run 'prepare' first")
        sys.exit(1)

    print("=" * 70)
    print(f"SUBMIT QUALITY SCORING BATCHES — {dataset_name}")
    print("=" * 70)

    submissions = {}

    for tagger_key in tagger_keys:
        tagger = TAGGERS[tagger_key]
        provider = tagger["provider"]

        for cond in conditions:
            prefix = f"quality_{cond}_{tagger_key}"
            batch_files = sorted(job_dir.glob(f"{prefix}_batch_*.jsonl"))
            if not batch_files:
                print(f"  {prefix}: no batch files found")
                continue

            print(f"\n  {tagger['label']} / Condition {cond} ({provider}): {len(batch_files)} batch(es)")

            for batch_file in batch_files:
                batch_name = batch_file.stem
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
                        "tagger": tagger_key,
                        "condition": cond,
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

    # Update manifest
    manifest_path = job_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "submitted"
    manifest["submissions"] = submissions
    manifest["submitted_at"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Next: python batch_quality_score.py status --dataset {dataset_name}")


def _submit_anthropic_batch(batch_file: Path) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    requests = []
    with open(batch_file) as f:
        for line in f:
            requests.append(json.loads(line))
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def _submit_openai_batch(batch_file: Path) -> str:
    from openai import OpenAI
    client = OpenAI()
    with open(batch_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


# ── Stage: status ─────────────────────────────────────────────────────

def cmd_status(args):
    """Check status of submitted batches."""
    dataset_name = args.dataset
    job_dir = get_job_dir(dataset_name)
    conditions = [args.condition] if args.condition else CONDITIONS
    tagger_keys = [args.tagger] if args.tagger else list(TAGGERS.keys())

    all_done = True
    any_found = False

    print("=" * 70)
    print(f"BATCH STATUS — {dataset_name} quality scoring")
    print("=" * 70)

    for tagger_key in tagger_keys:
        for cond in conditions:
            prefix = f"quality_{cond}_{tagger_key}"
            status_files = sorted(job_dir.glob(f"{prefix}_batch_*.status.json"))
            if not status_files:
                continue
            any_found = True

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
                    print(f"  {batch_name}: ERROR — {e}")
                    all_done = False

    if not any_found:
        print("  No batches found. Run 'submit' first.")
        return

    if all_done:
        print(f"\n  All batches complete!")
        print(f"  Next: python batch_quality_score.py download --dataset {dataset_name}")
    else:
        print(f"\n  Some batches still running. Check again later.")


def _check_anthropic_batch(batch_id: str) -> dict:
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
    }


def _check_openai_batch(batch_id: str) -> dict:
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


# ── Stage: download ───────────────────────────────────────────────────

def cmd_download(args):
    """Download completed batch results."""
    dataset_name = args.dataset
    job_dir = get_job_dir(dataset_name)
    results_dir = job_dir / "raw_results"
    results_dir.mkdir(exist_ok=True)
    conditions = [args.condition] if args.condition else CONDITIONS
    tagger_keys = [args.tagger] if args.tagger else list(TAGGERS.keys())

    print("=" * 70)
    print(f"DOWNLOAD QUALITY SCORING RESULTS — {dataset_name}")
    print("=" * 70)

    for tagger_key in tagger_keys:
        for cond in conditions:
            prefix = f"quality_{cond}_{tagger_key}"
            status_files = sorted(job_dir.glob(f"{prefix}_batch_*.status.json"))
            if not status_files:
                continue

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

    print(f"\n  Next: python batch_quality_score.py merge --dataset {dataset_name}")


def _download_anthropic_results(batch_id: str) -> list[dict]:
    from anthropic import Anthropic
    client = Anthropic()
    results = []
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            content = result.result.message.content
            text = content[0].text if content and len(content) > 0 else ""
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


# ── Stage: merge ──────────────────────────────────────────────────────

VALID_QUALITIES = {"good", "acceptable", "poor", "critical"}
VALID_VISIBILITIES = {"none", "visible", "invisible", "mixed"}
VALID_ARCHETYPES = {
    "the_confidence_trap", "the_silent_mismatch", "the_drift",
    "the_death_spiral", "the_contradiction_unravel", "the_walkaway",
    "the_partial_recovery", "the_mystery_failure", "none",
}


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from a model response."""
    import re
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... }
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _validate_response(parsed: dict) -> bool:
    if not parsed:
        return False
    if parsed.get("quality") not in VALID_QUALITIES:
        return False
    if parsed.get("failure_visibility") not in VALID_VISIBILITIES:
        return False
    archetypes = parsed.get("archetypes", [])
    if not isinstance(archetypes, list) or len(archetypes) == 0:
        return False
    return True


def _normalize_response(parsed: dict) -> dict:
    for a in parsed.get("archetypes", []):
        name = a["archetype"].lower().strip().replace(" ", "_")
        if not name.startswith("the_") and name not in ("none", "visible_failure"):
            name = "the_" + name
        a["archetype"] = name
    return parsed


def cmd_merge(args):
    """Parse quality scoring results into unified output files."""
    dataset_name = args.dataset
    job_dir = get_job_dir(dataset_name)
    results_dir = job_dir / "raw_results"
    output_dir = SCRIPT_DIR / "data" / "results_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [args.condition] if args.condition else CONDITIONS
    tagger_keys = [args.tagger] if args.tagger else list(TAGGERS.keys())

    print("=" * 70)
    print(f"MERGE QUALITY SCORING RESULTS — {dataset_name}")
    print("=" * 70)

    for tagger_key in tagger_keys:
        for cond in conditions:
            prefix = f"quality_{cond}_{tagger_key}"
            result_files = sorted(results_dir.glob(f"{prefix}_batch_*.results.jsonl"))

            if not result_files:
                print(f"\n  {prefix}: no result files")
                continue

            print(f"\n  {TAGGERS[tagger_key]['label']} / Condition {cond}:")

            records = []
            n_ok = 0
            n_parse_err = 0
            n_api_err = 0

            for rf in result_files:
                with open(rf) as f:
                    for line in f:
                        r = json.loads(line)
                        custom_id = r["custom_id"]

                        # Parse custom_id: "quality_{cond}_{cid}"
                        parts = custom_id.split("_", 2)
                        cid = int(parts[2]) if parts[2].isdigit() else parts[2]

                        if r["status"] != "ok":
                            n_api_err += 1
                            records.append({
                                "conversation_id": cid,
                                "condition": cond,
                                "tagger": tagger_key,
                                "status": "api_error",
                                "error": r.get("error", "unknown"),
                            })
                            continue

                        parsed = _parse_json_response(r["response"])
                        if parsed and _validate_response(parsed):
                            parsed = _normalize_response(parsed)
                            n_ok += 1
                            records.append({
                                "conversation_id": cid,
                                "condition": cond,
                                "tagger": tagger_key,
                                "quality": parsed["quality"],
                                "quality_rationale": parsed.get("quality_rationale", ""),
                                "failure_visibility": parsed["failure_visibility"],
                                "failure_visibility_rationale": parsed.get("failure_visibility_rationale", ""),
                                "archetypes": parsed["archetypes"],
                                "archetype_count": len(parsed["archetypes"]),
                                "raw_response": r["response"],
                                "status": "ok",
                            })
                        else:
                            n_parse_err += 1
                            records.append({
                                "conversation_id": cid,
                                "condition": cond,
                                "tagger": tagger_key,
                                "status": "parse_error",
                                "raw_response": r.get("response", ""),
                            })

            # Write output
            output_file = output_dir / f"{dataset_name}_quality_{cond}_{tagger_key}.jsonl"
            with open(output_file, "w") as f:
                for rec in sorted(records, key=lambda x: x["conversation_id"]):
                    f.write(json.dumps(rec) + "\n")

            print(f"    Results: {n_ok} ok, {n_parse_err} parse errors, {n_api_err} API errors")
            print(f"    → {output_file.name}")

            # Quality + visibility distribution for ok results
            if n_ok > 0:
                ok_recs = [r for r in records if r.get("status") == "ok"]
                total = len(ok_recs)

                q_dist = Counter(r["quality"] for r in ok_recs)
                dist_str = ", ".join(f"{k}: {v} ({v/total*100:.1f}%)"
                                     for k, v in sorted(q_dist.items()))
                print(f"    Quality distribution: {dist_str}")

                v_dist = Counter(r["failure_visibility"] for r in ok_recs)
                vis_str = ", ".join(f"{k}: {v} ({v/total*100:.1f}%)"
                                    for k, v in sorted(v_dist.items()))
                print(f"    Visibility distribution: {vis_str}")

    print(f"\n{'=' * 70}")
    print(f"DONE — Results in data/results_quality/")
    print(f"{'=' * 70}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch quality-scoring pipeline for transcript classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conditions:
    A — Transcript only (no signals)
    B — Transcript + inline turn-level signals + signal summary
    C — Signal profile only (no transcript text)

Example workflow:
    python batch_quality_score.py prepare  --dataset score_10k
    python batch_quality_score.py submit   --dataset score_10k
    python batch_quality_score.py status   --dataset score_10k
    python batch_quality_score.py download --dataset score_10k
    python batch_quality_score.py merge    --dataset score_10k

Filter by condition or tagger:
    python batch_quality_score.py prepare --dataset score_10k --condition B --tagger opus
""",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    def add_common_args(p):
        p.add_argument("--dataset", required=True,
                       help="Dataset name (e.g. score_10k, score_30k)")
        p.add_argument("--condition", choices=CONDITIONS, default=None,
                       help="Run only this condition (default: both B and C)")
        p.add_argument("--tagger", choices=list(TAGGERS.keys()), default=None,
                       help="Run only this tagger (default: all)")
        p.add_argument("--scheme", choices=["baseline", "calibrated"],
                       default=DEFAULT_SCHEME,
                       help="Signal scheme to use (default: calibrated)")

    p_prep = subparsers.add_parser("prepare", help="Build batch request files")
    add_common_args(p_prep)
    p_prep.add_argument("--dry-run", action="store_true")

    p_sub = subparsers.add_parser("submit", help="Submit batches to APIs")
    add_common_args(p_sub)

    p_stat = subparsers.add_parser("status", help="Check batch status")
    add_common_args(p_stat)

    p_dl = subparsers.add_parser("download", help="Download completed results")
    add_common_args(p_dl)

    p_merge = subparsers.add_parser("merge", help="Merge results")
    add_common_args(p_merge)

    args = parser.parse_args()

    {"prepare": cmd_prepare, "submit": cmd_submit, "status": cmd_status,
     "download": cmd_download, "merge": cmd_merge}[args.stage](args)


if __name__ == "__main__":
    main()
