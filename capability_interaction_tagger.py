#!/usr/bin/env python3
"""
Capability-vs-Interaction Layer Tagging Script

Takes a stratified sample of failure transcripts from the WildChat dataset,
sends each to Claude Opus for analysis of whether the failure is primarily
capability-layer, interaction-layer, or both.

Usage:
    # Dry run (5 transcripts, no API calls, prints prompts):
    python capability_interaction_tagger.py --dry-run

    # Dry run with actual API calls on 5 transcripts:
    python capability_interaction_tagger.py --dry-run --call-api

    # Full run with 20 parallel workers:
    python capability_interaction_tagger.py --workers 20

    # Custom sample size:
    python capability_interaction_tagger.py --sample-size 500 --workers 20

Requires:
    ANTHROPIC_API_KEY environment variable set (or in .env file)
    pip install anthropic pandas python-dotenv
"""

import argparse
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on environment variables directly
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse the project's existing signal/archetype infrastructure
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
import techreport_utils


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

ARCHETYPE_ORDER = [
    'Visible failure',
    'The confidence trap',
    'The silent mismatch',
    'The drift',
    'The death spiral',
    'The contradiction unravel',
    'The walkaway',
    'The partial recovery',
    'The mystery failure',
]


def primary_archetype(archetypes: list[str]) -> str:
    """Assign a single primary archetype for stratification purposes."""
    for a in ARCHETYPE_ORDER:
        if a in archetypes:
            return a
    return archetypes[0] if archetypes else 'Unknown'


def build_sample(
    df: pd.DataFrame,
    total_n: int = 1050,
    min_per_archetype: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified sample of failure transcripts.

    Strategy:
    - Min 100 per archetype (or all available if fewer than 100)
    - Remaining slots filled proportionally
    - Includes ~100-125 'Visible failure' as a comparison group
    """
    fail_df = df[df.signals.apply(lambda x: 'goal_failure' in x)].copy()
    fail_df['primary_archetype'] = fail_df.archetypes.apply(primary_archetype)

    # Show distribution
    dist = fail_df['primary_archetype'].value_counts()
    print("Archetype distribution in full failure set:")
    for arch, count in dist.items():
        print(f"  {arch}: {count:,}")
    print(f"  Total: {fail_df.shape[0]:,}")
    print()

    rng = random.Random(random_state)
    sampled_ids = set()
    samples_by_arch = {}

    # Phase 1: guarantee minimum per archetype
    for arch in ARCHETYPE_ORDER:
        arch_df = fail_df[fail_df.primary_archetype == arch]
        n_available = arch_df.shape[0]
        n_take = min(min_per_archetype, n_available)
        chosen = arch_df.sample(n=n_take, random_state=random_state)
        samples_by_arch[arch] = chosen.index.tolist()
        sampled_ids.update(chosen.index.tolist())

    # Phase 2: fill remaining slots proportionally from un-sampled transcripts
    remaining_budget = total_n - len(sampled_ids)
    if remaining_budget > 0:
        unsampled = fail_df[~fail_df.index.isin(sampled_ids)]
        if not unsampled.empty:
            # Proportional to archetype frequency
            arch_weights = unsampled.primary_archetype.value_counts(normalize=True)
            for arch in ARCHETYPE_ORDER:
                if arch not in arch_weights:
                    continue
                n_extra = int(remaining_budget * arch_weights[arch])
                arch_unsampled = unsampled[unsampled.primary_archetype == arch]
                n_take = min(n_extra, arch_unsampled.shape[0])
                if n_take > 0:
                    chosen = arch_unsampled.sample(n=n_take, random_state=random_state)
                    samples_by_arch[arch].extend(chosen.index.tolist())
                    sampled_ids.update(chosen.index.tolist())

    sample_df = fail_df.loc[list(sampled_ids)].copy()

    print(f"Sample drawn: {sample_df.shape[0]} transcripts")
    print("Sample distribution:")
    for arch, count in sample_df.primary_archetype.value_counts().items():
        print(f"  {arch}: {count}")
    print()

    return sample_df


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert analyst studying failures in human-AI interactions. You are reviewing transcripts from the WildChat dataset (GPT-4, 2023-2024) that have been tagged with quality signals and failure archetypes by an upstream classifier.

Your task is to analyze each transcript through two independent lenses:

1. THE CAPABILITY LAYER: Did the model lack the knowledge, reasoning ability, or skill to perform the task correctly? This is strictly about whether the model COULD do the task — did it have the factual knowledge, the reasoning ability, the code generation skill? Examples: factual errors, buggy code, mathematical mistakes, hallucinated content, inability to perform a complex reasoning chain.

2. THE INTERACTION/BEHAVIOR LAYER: Did the failure arise from how the model BEHAVED in the interaction, as distinct from what it was capable of? This is about the model's behavioral choices and the dynamics of the human-AI exchange. Even a model that is perfectly capable of performing a task can fail at the interaction layer if it:
   (a) Encounters genuine ambiguity or underspecification in the user's request and guesses rather than asking for clarification — this is a behavioral choice shaped by training incentives (RLHF rewards answering over clarifying), not a capability limitation
   (b) Presents its output with confidence and fluency that masks problems from the user — the model may "know" it is uncertain but still presents confidently because that's how it was trained to behave
   (c) Generates content immediately rather than asking for necessary context (height, weight, preferences, constraints) — the "generate rather than clarify" default is a product/behavior pattern, not an intelligence limitation
   (d) Operates within a product surface that doesn't support what the user needs (no web access, no image processing, no persistent memory, output length limits)
   (e) Encounters user formatting, phrasing, or cultural conventions that create parsing ambiguity

CRITICAL DISTINCTION: The interaction/behavior layer is NOT about whether a smarter model could "figure it out." A model can be smart enough to recognize ambiguity but still trained to guess rather than ask. A model can be capable of flagging uncertainty but still behaviorally default to confident presentation. The question is whether the failure involves dynamics of how the model behaves in the interaction and how the product mediates the exchange — dynamics that are shaped by training incentives, product design, and the inherent properties of human communication, not by raw model intelligence.

These two layers are INDEPENDENT. A failure can have both a capability component (the model got the answer wrong) AND an interaction component (it presented the wrong answer with false confidence). Assess each layer separately.

You must respond with valid JSON matching the schema described in each request."""


def build_user_prompt(row: pd.Series) -> str:
    """Build the per-transcript analysis prompt."""

    # Truncate very long transcripts to avoid token overflow
    transcript = row['transcript']
    if len(transcript) > 12000:
        transcript = transcript[:12000] + "\n\n[... TRANSCRIPT TRUNCATED FOR LENGTH ...]"

    signals_list = row['signals']
    archetypes_list = row['archetypes']
    quality = row['overall_quality']

    return f"""Analyze the following human-AI transcript that has been flagged as a failure case.

## Upstream annotations (from a prior classifier)
- Overall quality: {quality}
- Quality signals: {json.dumps(signals_list)}
- Failure archetypes: {json.dumps(archetypes_list)}

## Transcript
<transcript>
{transcript}
</transcript>

## Your analysis

Respond with a JSON object with exactly these fields:

{{
  "validation": {{
    "is_actual_failure": "confirmed" | "plausible" | "questionable" | "disagree",
    "validation_notes": "Brief explanation of whether this transcript actually contains a meaningful failure."
  }},
  "capability_layer": {{
    "has_capability_gap": true | false,
    "capability_notes": "Did the model lack the knowledge, reasoning ability, or skill to do this task correctly? Focus strictly on whether it COULD do the task — not how it chose to behave. Explain.",
    "specific_capability_gaps": ["List specific capability deficiencies, e.g., 'incorrect factual claim about X', 'buggy code logic in Y', 'mathematical error in Z'. Empty list if none."]
  }},
  "interaction_layer": {{
    "interaction_dynamics_present": ["List which interaction dynamics apply from: 'ambiguity_underspecification', 'false_confidence_masking', 'product_capability_mismatch', 'missing_verification_mechanism', 'formatting_encoding_issue', 'generate_rather_than_clarify'. Empty list if none."],
    "interaction_notes": "Explain how the model's behavioral choices and/or the interaction dynamics contributed to the failure or made it invisible.",
    "would_persist_with_better_model": "definitely_yes" | "probably_yes" | "uncertain" | "probably_no" | "definitely_no",
    "persistence_notes": "Would these interaction/behavioral dynamics persist even with a much more capable model? Remember: a smarter model that is still trained to answer rather than clarify, still trained to present confidently, still operating in a chat UI without verification mechanisms — that model still has the same interaction-layer dynamics. Only answer 'probably_no' if the behavioral pattern itself would change, not just because a smarter model might happen to get the right answer despite the pattern."
  }},
  "overall_classification": {{
    "primary_layer": "capability" | "interaction" | "both" | "unclear",
    "rationale": "One sentence explaining the classification."
  }}
}}

Important:
- has_capability_gap asks: did the model lack the SKILL or KNOWLEDGE to do this? Pure capability question.
- would_persist_with_better_model asks: would the BEHAVIORAL/INTERACTION dynamics persist? This is NOT asking "would a smarter model get the right answer" — it's asking "would the interaction pattern (guessing instead of clarifying, presenting confidently, generating rather than asking for context) still be present even with a more capable model?"
- These are independent. A math error (capability) can co-occur with false confidence presentation (interaction). Both matter.
- Be precise about specific capability gaps. "The model hallucinated X" is good. "The model was bad" is not.
- Example: A user asks a vague question and the model gives a generic answer instead of asking for specifics. Even if a smarter model gives a BETTER generic answer, the generate-rather-than-clarify dynamic persists. That's interaction-layer.
"""


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------

def call_opus(
    transcript_row: pd.Series,
    client,
    model: str = "claude-opus-4-20250514",
    max_retries: int = 3,
) -> dict:
    """Call Claude Opus to analyze a single transcript. Returns the parsed response."""
    user_prompt = build_user_prompt(transcript_row)
    conversation_id = transcript_row['conversation_id']

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )

            text = response.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            parsed['conversation_id'] = conversation_id
            parsed['_raw_response'] = text
            parsed['_input_tokens'] = response.usage.input_tokens
            parsed['_output_tokens'] = response.usage.output_tokens
            return parsed

        except json.JSONDecodeError as e:
            print(f"  [conv {conversation_id}] JSON parse error on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return {
                    'conversation_id': conversation_id,
                    '_error': f'JSON parse error: {e}',
                    '_raw_response': text if 'text' in dir() else None,
                }
        except Exception as e:
            wait_time = 2 ** attempt + random.random()
            print(f"  [conv {conversation_id}] API error on attempt {attempt+1}: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                return {
                    'conversation_id': conversation_id,
                    '_error': str(e),
                }

    return {'conversation_id': conversation_id, '_error': 'Max retries exceeded'}


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def load_and_prepare_data() -> pd.DataFrame:
    """Load the tagged CSV and prepare signals/archetypes exactly as the notebook does."""
    script_dir = Path(__file__).parent

    # Load tagged data
    tag_df = pd.read_csv(script_dir / "wildchat_big_ux_tagged.csv")
    tag_df = tag_df.rename(columns={
        "full_response_json": "full_tagging_response_json",
        "model": "tagging_model"
    })

    # Optionally merge use-case data if available (provides primary_vertical)
    use_case_path = script_dir / "r.csv"
    if not use_case_path.exists():
        # Try alternate name
        use_case_path = script_dir / "wildchat_big_use_cases.csv"

    if use_case_path.exists():
        use_df = pd.read_csv(use_case_path)
        use_df = use_df.rename(columns={
            "full_response_json": "full_use_case_response_json",
            "model": "use_case_model"
        })
        df = pd.merge(tag_df, use_df, on="conversation_id", how="outer")
        print(f"  Merged use-case data from {use_case_path.name}")
    else:
        df = tag_df.copy()
        print("  Note: use-case CSV (r.csv) not found; proceeding without domain labels.")
        # Add placeholder columns so downstream code doesn't break
        df['primary_vertical'] = None
        df['secondary_vertical'] = None

    # Exclude invalid_input
    df = df[df.signals_json.str.contains("invalid_input") == False].copy()

    # Compute signals and archetypes (matching notebook exactly)
    df['signals'] = df.signals_json.apply(techreport_utils.get_signals)
    df['archetypes'] = df.signals.apply(techreport_utils.assign_archetypes)

    print(f"Loaded {df.shape[0]:,} transcripts (after excluding invalid_input)")
    return df


def run_dry_run(sample_df: pd.DataFrame, call_api: bool = False):
    """Print example prompts and optionally call API on a few."""
    print("=" * 80)
    print("DRY RUN: Showing 5 example prompts")
    print("=" * 80)

    examples = sample_df.head(5)

    for i, (idx, row) in enumerate(examples.iterrows()):
        print(f"\n{'─' * 80}")
        print(f"Example {i+1}: conversation_id={row['conversation_id']}")
        print(f"  quality={row['overall_quality']}, archetype={row['primary_archetype']}")
        print(f"  signals={row['signals']}")
        print(f"  transcript length={len(row['transcript'])} chars")
        print(f"{'─' * 80}")

        prompt = build_user_prompt(row)
        # Show first 2000 chars of prompt
        print(prompt[:2000])
        if len(prompt) > 2000:
            print(f"\n  ... [{len(prompt) - 2000} more chars] ...")
        print()

    if call_api:
        print("\n" + "=" * 80)
        print("DRY RUN: Calling API on 5 examples")
        print("=" * 80)

        import anthropic
        client = anthropic.Anthropic()

        for i, (idx, row) in enumerate(examples.iterrows()):
            print(f"\n  Calling Opus for conversation_id={row['conversation_id']}...")
            result = call_opus(row, client)

            if '_error' in result:
                print(f"  ERROR: {result['_error']}")
            else:
                print(f"  Tokens: {result.get('_input_tokens', '?')} in, {result.get('_output_tokens', '?')} out")
                # Pretty-print the result (without raw response)
                display = {k: v for k, v in result.items() if not k.startswith('_')}
                print(json.dumps(display, indent=2))

    print("\n" + "=" * 80)
    print("DRY RUN COMPLETE")
    print(f"To run the full analysis on {sample_df.shape[0]} transcripts:")
    print(f"  python {Path(__file__).name} --workers 20")
    print("=" * 80)


def run_full(sample_df: pd.DataFrame, workers: int, output_path: Path):
    """Run the full parallel analysis."""
    import anthropic
    client = anthropic.Anthropic()

    total = sample_df.shape[0]
    print(f"Starting full run: {total} transcripts, {workers} workers")
    print(f"Output: {output_path}")
    print()

    results = []
    completed = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, row in sample_df.iterrows():
            future = executor.submit(call_opus, row, client)
            futures[future] = row['conversation_id']

        for future in as_completed(futures):
            conv_id = futures[future]
            try:
                result = future.result()
                results.append(result)

                if '_error' in result:
                    errors += 1
                else:
                    total_input_tokens += result.get('_input_tokens', 0)
                    total_output_tokens += result.get('_output_tokens', 0)

                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0

                if completed % 25 == 0 or completed == total:
                    est_remaining = (total - completed) / rate if rate > 0 else 0
                    print(
                        f"  [{completed}/{total}] "
                        f"{rate:.1f}/s, "
                        f"~{est_remaining:.0f}s remaining, "
                        f"{errors} errors, "
                        f"tokens: {total_input_tokens:,}in/{total_output_tokens:,}out"
                    )

            except Exception as e:
                print(f"  FATAL error for conv_id={conv_id}: {e}")
                results.append({'conversation_id': conv_id, '_error': str(e)})
                errors += 1
                completed += 1

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({completed} transcripts, {errors} errors)")
    print(f"Total tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")

    # Estimate cost (Opus pricing: $15/M input, $75/M output)
    est_cost = (total_input_tokens / 1_000_000 * 15) + (total_output_tokens / 1_000_000 * 75)
    print(f"Estimated cost: ${est_cost:.2f}")

    # Save results
    # Save full results as JSON (preserves nested structure)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    # Also save a flat CSV for quick analysis
    flat_rows = []
    for r in results:
        if '_error' in r:
            flat_rows.append({
                'conversation_id': r['conversation_id'],
                'error': r['_error'],
            })
            continue

        flat_rows.append({
            'conversation_id': r['conversation_id'],
            'validation': r.get('validation', {}).get('is_actual_failure'),
            'validation_notes': r.get('validation', {}).get('validation_notes'),
            'has_capability_gap': r.get('capability_layer', {}).get('has_capability_gap'),
            'capability_notes': r.get('capability_layer', {}).get('capability_notes'),
            'specific_capability_gaps': json.dumps(r.get('capability_layer', {}).get('specific_capability_gaps', [])),
            'interaction_dynamics': json.dumps(r.get('interaction_layer', {}).get('interaction_dynamics_present', [])),
            'interaction_notes': r.get('interaction_layer', {}).get('interaction_notes'),
            'would_persist': r.get('interaction_layer', {}).get('would_persist_with_better_model'),
            'persistence_notes': r.get('interaction_layer', {}).get('persistence_notes'),
            'primary_layer': r.get('overall_classification', {}).get('primary_layer'),
            'rationale': r.get('overall_classification', {}).get('rationale'),
        })

    flat_df = pd.DataFrame(flat_rows)
    csv_path = output_path.with_suffix('.csv')
    flat_df.to_csv(csv_path, index=False)
    print(f"Flat CSV saved to {csv_path}")

    # Print quick summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    if 'primary_layer' in flat_df.columns:
        print("\nOverall classification:")
        print(flat_df['primary_layer'].value_counts().to_string())

        print("\nValidation:")
        print(flat_df['validation'].value_counts().to_string())

        print("\nHas capability gap?:")
        print(flat_df['has_capability_gap'].value_counts().to_string())

        print("\nWould interaction dynamics persist with better model?:")
        print(flat_df['would_persist'].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Capability vs Interaction layer tagging")
    parser.add_argument('--dry-run', action='store_true', help='Show example prompts without running full analysis')
    parser.add_argument('--call-api', action='store_true', help='In dry-run mode, also call API on examples')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel workers (default: 20)')
    parser.add_argument('--sample-size', type=int, default=1050, help='Total sample size (default: 1050)')
    parser.add_argument('--min-per-archetype', type=int, default=100, help='Minimum samples per archetype (default: 100)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    print("Loading data...")
    df = load_and_prepare_data()

    print("Building stratified sample...")
    sample_df = build_sample(
        df,
        total_n=args.sample_size,
        min_per_archetype=args.min_per_archetype,
        random_state=args.seed,
    )

    if args.dry_run:
        run_dry_run(sample_df, call_api=args.call_api)
    else:
        if not os.environ.get('ANTHROPIC_API_KEY'):
            print("ERROR: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)

        output_path = Path(args.output) if args.output else Path(__file__).parent / "capability_interaction_results.json"
        run_full(sample_df, args.workers, output_path)


if __name__ == '__main__':
    main()
