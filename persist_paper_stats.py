#!/usr/bin/env python3
"""Step 7: Compute the exact stats reported in the paper's persistence section.

Reproduces the numbers from bigspin-invisible-failures.tex §5 (persistence
experiment) using the v2 2K dataset, separately for Sonnet and GPT-4.1.

Stats computed:
  1. Overall failure rates per model (Fig failure-persist)
     - % none / visible / invisible / mixed
     - Overall failure rate, invisible failure rate within failures
  2. Archetype distribution per model (Fig arch-persist)
     - Restricted to invisible+mixed failures, normalized to % of that total
  3. Problem signal rate deltas (Tab persist-signals)
     - Top 5 most reduced, top 5 most persistent (n≥5 filter)
     - n, %, Δ pp, % Eliminated

Usage:
    python 07_paper_stats.py
    python 07_paper_stats.py --condition C --tagger opus --scheme calibrated
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent.parent  # production-turn-tagging/
RESULTS_DIR = PARENT_DIR / "data" / "results"
RESULTS_QUALITY_DIR = PARENT_DIR / "data" / "results_quality"

PROBLEM_SIGNALS = {
    "factual_error", "false_confidence", "performative_hedge", "silent_assumption",
    "generate_without_clarifying", "repetition", "plow_through", "error_commitment",
    "over_delivered", "conversation_stalled", "ai_implicit_refusal", "ethical_tension",
    "off_topic_drift", "intent_missed", "under_delivered", "problem_ignored",
    "user_misled", "ai_self_contradiction", "ai_malfunction",
}


# ── Data loading ─────────────────────────────────────────────────────────


def load_sample_cids() -> set:
    sample_path = SCRIPT_DIR / "data" / "sample" / "sample_2k.jsonl"
    cids = set()
    with open(sample_path) as f:
        for line in f:
            cids.add(json.loads(line)["conversation_id"])
    return cids


def load_quality_results(dataset_name: str, condition: str, tagger: str) -> dict:
    path = RESULTS_QUALITY_DIR / f"{dataset_name}_quality_{condition}_{tagger}.jsonl"
    results = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("status") == "ok":
            results[r["conversation_id"]] = r
    return results


def load_turns(path: Path) -> list[dict]:
    turns = []
    with open(path) as f:
        for line in f:
            turns.append(json.loads(line))
    return turns


def get_archetypes(rec: dict) -> set[str]:
    archetypes = set()
    for a in rec.get("archetypes", []):
        name = a.get("archetype", "none")
        if name != "none":
            archetypes.add(name)
    return archetypes


def signal_counts(turns: list[dict]) -> Counter:
    counts = Counter()
    for t in turns:
        seen = set()
        for s in t.get("ai_signals", []):
            name = s.get("signal", s) if isinstance(s, dict) else s
            seen.add(name)
        for sig in seen:
            counts[sig] += 1
    return counts


# ── Stat 1: Failure rates (Fig failure-persist) ─────────────────────────


def compute_failure_rates(data: dict, cids: list, label: str) -> dict:
    """Compute failure visibility breakdown for one model."""
    n = len(cids)
    vis_counts = Counter()
    for cid in cids:
        q = data[cid].get("quality", "good")
        vis = data[cid].get("failure_visibility", "none")
        if q == "good":
            vis_counts["none"] += 1
        else:
            vis_counts[vis] += 1

    n_failure = n - vis_counts["none"]
    n_invisible = vis_counts.get("invisible", 0)
    n_mixed = vis_counts.get("mixed", 0)
    n_visible = vis_counts.get("visible", 0)

    failure_rate = n_failure / n
    invis_of_failures = (n_invisible + n_mixed) / n_failure if n_failure > 0 else 0

    return {
        "label": label, "n": n,
        "none": vis_counts["none"],
        "visible": n_visible,
        "invisible": n_invisible,
        "mixed": n_mixed,
        "n_failure": n_failure,
        "failure_rate": failure_rate,
        "invis_of_failures": invis_of_failures,
    }


def print_failure_rates(stats_list: list[dict]):
    """Print the failure rate table matching Fig failure-persist."""
    print(f"\n{'=' * 80}")
    print(f"  FAILURE RATES (Fig failure-persist equivalent)")
    print(f"{'=' * 80}\n")

    print(f"  {'Model':<12s} {'N':>5s}  {'None':>10s}  {'Visible':>10s}  "
          f"{'Invisible':>10s}  {'Mixed':>10s}  {'Fail %':>7s}  {'Invis/Fail':>10s}")
    print(f"  {'-' * 85}")

    for s in stats_list:
        n = s["n"]
        print(f"  {s['label']:<12s} {n:>5}  "
              f"{s['none']:>4} ({s['none']/n:>5.1%})  "
              f"{s['visible']:>4} ({s['visible']/n:>5.1%})  "
              f"{s['invisible']:>4} ({s['invisible']/n:>5.1%})  "
              f"{s['mixed']:>4} ({s['mixed']/n:>5.1%})  "
              f"{s['failure_rate']:>6.1%}  "
              f"{s['invis_of_failures']:>9.1%}")

    # Print the key sentence the paper uses
    print(f"\n  Paper sentence: \"failure rates have gone down substantially, "
          f"from {stats_list[0]['failure_rate']:.1%} to between "
          f"{min(s['failure_rate'] for s in stats_list[1:]):.1%} and "
          f"{max(s['failure_rate'] for s in stats_list[1:]):.1%} "
          f"for today's best models.\"")


# ── Stat 2: Archetype distribution (Fig arch-persist) ───────────────────


def compute_archetype_dist(data: dict, cids: list, label: str) -> dict:
    """Archetype distribution restricted to invisible+mixed failures."""
    invis_cids = []
    for cid in cids:
        vis = data[cid].get("failure_visibility", "none")
        if vis in ("invisible", "mixed"):
            invis_cids.append(cid)

    n_invis = len(invis_cids)
    arch_counts = Counter()
    for cid in invis_cids:
        for a in get_archetypes(data[cid]):
            arch_counts[a] += 1

    return {
        "label": label,
        "n_invis": n_invis,
        "arch_counts": arch_counts,
    }


def print_archetype_dist(dist_list: list[dict]):
    """Print archetype distribution table matching Fig arch-persist."""
    print(f"\n{'=' * 80}")
    print(f"  ARCHETYPE DISTRIBUTION — INVISIBLE+MIXED FAILURES (Fig arch-persist equivalent)")
    print(f"{'=' * 80}\n")

    # Collect all archetypes
    all_archs = set()
    for d in dist_list:
        all_archs |= set(d["arch_counts"].keys())

    # Sort by original prevalence
    ref = dist_list[0]
    sorted_archs = sorted(all_archs, key=lambda a: -ref["arch_counts"].get(a, 0))

    # Header
    header = f"  {'Archetype':<30s}"
    for d in dist_list:
        header += f"  {d['label']:>18s}"
    print(header)
    print(f"  {'-' * (30 + 20 * len(dist_list))}")

    for arch in sorted_archs:
        row = f"  {arch:<30s}"
        for d in dist_list:
            c = d["arch_counts"].get(arch, 0)
            pct = c / d["n_invis"] * 100 if d["n_invis"] > 0 else 0
            row += f"  {c:>4} ({pct:>5.1f}%){' ':>6s}"
        print(row)

    # Totals
    print()
    for d in dist_list:
        print(f"  {d['label']}: {d['n_invis']} invisible+mixed failures")


# ── Stat 3: Signal rate deltas (Tab persist-signals) ────────────────────


def compute_signal_deltas(
    orig_turns: list[dict], new_turns: list[dict], label: str, min_n: int = 5
) -> list[dict]:
    """Compute per-signal rate deltas (Tab persist-signals format)."""
    orig_n = len(orig_turns)
    new_n = len(new_turns)
    orig_c = signal_counts(orig_turns)
    new_c = signal_counts(new_turns)

    all_sigs = sorted(set(orig_c.keys()) | set(new_c.keys()))

    results = []
    for sig in all_sigs:
        if sig not in PROBLEM_SIGNALS:
            continue

        oc = orig_c.get(sig, 0)
        nc = new_c.get(sig, 0)

        # Filter: at least min_n in either
        if max(oc, nc) < min_n:
            continue

        o_rate = oc / orig_n * 100
        n_rate = nc / new_n * 100
        delta_pp = n_rate - o_rate

        if o_rate > 0:
            pct_elim = (o_rate - n_rate) / o_rate * 100
        else:
            pct_elim = None

        results.append({
            "signal": sig,
            "orig_n": oc, "new_n": nc,
            "orig_pct": o_rate, "new_pct": n_rate,
            "delta_pp": delta_pp,
            "pct_eliminated": pct_elim,
        })

    return results


def print_signal_table(results: list[dict], label: str, orig_n: int, new_n: int):
    """Print the signal table matching Tab persist-signals."""
    print(f"\n{'=' * 80}")
    print(f"  PROBLEM SIGNAL DELTAS: GPT-4 → {label} (Tab persist-signals equivalent)")
    print(f"  Original: {orig_n} turns, {label}: {new_n} turns")
    print(f"{'=' * 80}")

    by_elim = sorted(results, key=lambda x: -(x["pct_eliminated"] or -9999))

    print(f"\n  Top 5 most reduced (n≥5):")
    print(f"  {'Signal':<30s} {'Orig n':>7s} {'Orig %':>7s} {'New n':>7s} "
          f"{'New %':>7s} {'Δ pp':>8s} {'% Elim':>9s}")
    print(f"  {'-' * 76}")
    for r in by_elim[:5]:
        pe = f"{r['pct_eliminated']:.0f}" if r['pct_eliminated'] is not None else "n/a"
        print(f"  {r['signal']:<30s} {r['orig_n']:>7} {r['orig_pct']:>6.1f} "
              f"{r['new_n']:>7} {r['new_pct']:>6.1f} {r['delta_pp']:>+7.1f} {pe:>8s}")

    print(f"\n  Top 5 most persistent (n≥5):")
    print(f"  {'Signal':<30s} {'Orig n':>7s} {'Orig %':>7s} {'New n':>7s} "
          f"{'New %':>7s} {'Δ pp':>8s} {'% Elim':>9s}")
    print(f"  {'-' * 76}")
    for r in by_elim[-5:]:
        pe = f"{r['pct_eliminated']:.0f}" if r['pct_eliminated'] is not None else "n/a"
        print(f"  {r['signal']:<30s} {r['orig_n']:>7} {r['orig_pct']:>6.1f} "
              f"{r['new_n']:>7} {r['new_pct']:>6.1f} {r['delta_pp']:>+7.1f} {pe:>8s}")

    # Print LaTeX-ready version for copy-paste
    print(f"\n  LaTeX table rows (top 5 reduced + top 5 persistent):")
    print(f"  % GPT-4 → {label}")
    for r in by_elim[:5]:
        pe = f"{r['pct_eliminated']:.0f}" if r['pct_eliminated'] is not None else "n/a"
        sig_tex = r['signal'].replace('_', r'\_')
        print(f"    {sig_tex:<35s} & {r['orig_n']:>3} & {r['orig_pct']:.1f} "
              f"& {r['new_n']:>3} & {r['new_pct']:.1f} "
              f"& ${r['delta_pp']:+.1f}$ & {pe} \\\\")
    print(f"    \\midrule")
    for r in by_elim[-5:]:
        pe = f"{r['pct_eliminated']:.0f}" if r['pct_eliminated'] is not None else "n/a"
        sig_tex = r['signal'].replace('_', r'\_')
        print(f"    {sig_tex:<35s} & {r['orig_n']:>3} & {r['orig_pct']:.1f} "
              f"& {r['new_n']:>3} & {r['new_pct']:.1f} "
              f"& ${r['delta_pp']:+.1f}$ & {pe} \\\\")


# ── Comparison ──────────────────────────────────────────────────────────


def print_comparison(model_data: list[dict]):
    """Highlight key differences across all new models."""
    print(f"\n{'=' * 80}")
    labels = [m["label"] for m in model_data]
    print(f"  KEY DIFFERENCES: {' vs '.join(labels)}")
    print(f"{'=' * 80}\n")

    print(f"  Failure rates:")
    for m in model_data:
        print(f"    {m['label']:<12s}: {m['failure_rate']:.1%}")

    rates = [(m["label"], m["failure_rate"]) for m in model_data]
    best = min(rates, key=lambda x: x[1])
    worst = max(rates, key=lambda x: x[1])
    if best[1] < worst[1]:
        print(f"    → {best[0]} has lowest failure rate ({best[1]:.1%}), "
              f"{worst[0]} has highest ({worst[1]:.1%})")

    print(f"\n  Invisible failure rate (of failures):")
    for m in model_data:
        print(f"    {m['label']:<12s}: {m['invis_of_failures']:.1%}")

    # Signal comparison across all models
    print(f"\n  % Eliminated comparison across models:")
    sig_maps = {m["label"]: {r["signal"]: r for r in m["signals"]} for m in model_data}
    all_sigs = set()
    for sm in sig_maps.values():
        all_sigs |= set(sm.keys())

    # Header
    header = f"  {'Signal':<30s}"
    for label in labels:
        header += f"  {label:>12s}"
    header += f"  {'Max gap':>8s}"
    print(header)
    print(f"  {'-' * (30 + 14 * len(labels) + 10)}")

    rows = []
    for sig in all_sigs:
        vals = []
        for label in labels:
            r = sig_maps[label].get(sig)
            if r and r["pct_eliminated"] is not None:
                vals.append(r["pct_eliminated"])
            else:
                vals.append(None)
        non_none = [v for v in vals if v is not None]
        gap = max(non_none) - min(non_none) if len(non_none) >= 2 else 0
        rows.append((sig, vals, gap))

    rows.sort(key=lambda x: -x[2])
    for sig, vals, gap in rows[:10]:
        row = f"  {sig:<30s}"
        for v in vals:
            row += f"  {v:>11.0f}%" if v is not None else f"  {'n/a':>12s}"
        row += f"  {gap:>6.0f}pp"
        print(row)


# ── Main ────────────────────────────────────────────────────────────────

# Response models to load. Each entry: (dataset_key, label)
# "original" is always loaded from score_10k; the rest from persist2_*.
RESPONSE_MODELS = [
    ("sonnet",  "Sonnet"),
    ("gpt41",   "GPT-4.1"),
    ("opus",    "Opus"),
    ("gpt5",    "GPT-5.4"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Compute paper stats from v2 persistence experiment"
    )
    parser.add_argument("--condition", default="C", choices=["A", "B", "C"])
    parser.add_argument("--tagger", default="opus", choices=["opus", "gpt5"])
    parser.add_argument("--scheme", default="calibrated", choices=["calibrated", "baseline"])
    args = parser.parse_args()

    sample_cids = load_sample_cids()
    print(f"Sample: {len(sample_cids)} conversations")

    # ── Load quality data ──
    print("\nLoading quality/archetype results...")
    all_original_q = load_quality_results("score_10k", args.condition, "opus")
    original_q = {c: r for c, r in all_original_q.items() if c in sample_cids}
    print(f"  Original (score_10k): {len(original_q)}")

    model_quality = {}  # key -> quality dict
    available_models = []  # (key, label) for models with data
    for key, label in RESPONSE_MODELS:
        ds = f"persist2_{key}"
        path = RESULTS_QUALITY_DIR / f"{ds}_quality_{args.condition}_opus.jsonl"
        if not path.exists():
            print(f"  {label} ({ds}): NOT FOUND — skipping")
            continue
        q = load_quality_results(ds, args.condition, "opus")
        model_quality[key] = q
        available_models.append((key, label))
        print(f"  {label}: {len(q)}")

    # Shared conversations (original ∩ all available models)
    shared_set = set(original_q.keys())
    for key, _ in available_models:
        shared_set &= set(model_quality[key].keys())
    shared = sorted(sample_cids & shared_set)
    print(f"  Shared (across {1 + len(available_models)} models): {len(shared)}")

    # ── Load signal data ──
    tagger = args.tagger
    scheme = args.scheme
    print(f"\nLoading signal results (tagger={tagger}, scheme={scheme})...")

    all_orig_turns = load_turns(RESULTS_DIR / f"score_10k_{tagger}_{scheme}_turns.jsonl")
    orig_turns = [t for t in all_orig_turns if t["conversation_id"] in sample_cids and t["turn_number"] == 1]
    print(f"  Original: {len(orig_turns)} turns")

    model_turns = {}
    for key, label in available_models:
        path = RESULTS_DIR / f"persist2_{key}_{tagger}_{scheme}_turns.jsonl"
        if not path.exists():
            print(f"  {label}: turn file NOT FOUND — skipping signals")
            continue
        turns = load_turns(path)
        model_turns[key] = turns
        print(f"  {label}: {len(turns)} turns")

    # ═══════════════════════════════════════════════════════════════════════
    # STAT 1: Failure rates
    # ═══════════════════════════════════════════════════════════════════════

    orig_fr = compute_failure_rates(original_q, shared, "GPT-4")
    model_frs = {}
    for key, label in available_models:
        model_frs[key] = compute_failure_rates(model_quality[key], shared, label)
    print_failure_rates([orig_fr] + [model_frs[k] for k, _ in available_models])

    # ═══════════════════════════════════════════════════════════════════════
    # STAT 2: Archetype distribution
    # ═══════════════════════════════════════════════════════════════════════

    orig_ad = compute_archetype_dist(original_q, shared, "GPT-4")
    model_ads = {}
    for key, label in available_models:
        model_ads[key] = compute_archetype_dist(model_quality[key], shared, label)
    print_archetype_dist([orig_ad] + [model_ads[k] for k, _ in available_models])

    # ═══════════════════════════════════════════════════════════════════════
    # STAT 3: Signal rate deltas per model
    # ═══════════════════════════════════════════════════════════════════════

    model_sigs = {}
    for key, label in available_models:
        if key in model_turns:
            sig = compute_signal_deltas(orig_turns, model_turns[key], label)
            model_sigs[key] = sig
            print_signal_table(sig, label, len(orig_turns), len(model_turns[key]))

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON across all models
    # ═══════════════════════════════════════════════════════════════════════

    comparison_data = []
    for key, label in available_models:
        if key in model_frs and key in model_sigs:
            comparison_data.append({
                "label": label,
                "failure_rate": model_frs[key]["failure_rate"],
                "invis_of_failures": model_frs[key]["invis_of_failures"],
                "signals": model_sigs[key],
            })
    if len(comparison_data) >= 2:
        print_comparison(comparison_data)

    # ═══════════════════════════════════════════════════════════════════════
    # v1 NUMBERS (for comparison)
    # ═══════════════════════════════════════════════════════════════════════

    # v1 paper numbers (from .tex lines 260, 288-298)
    v1 = {
        "failure_rate_orig": 41.7,
        "failure_rate_range": (3.6, 13.8),
        "signals": {
            "intent_missed":        {"elim": 97},
            "performative_hedge":   {"elim": 97},
            "conversation_stalled": {"elim": 95},
            "problem_ignored":      {"elim": 92},
            "ai_implicit_refusal":  {"elim": 91},
            "silent_assumption":            {"elim": 48},
            "generate_without_clarifying":  {"elim": 38},
            "ethical_tension":              {"elim":  6},
            "ai_malfunction":              {"elim": -500},
            "over_delivered":              {"elim": -130},
        }
    }

    print(f"\n{'=' * 80}")
    print(f"  COMPARISON WITH PAPER's CURRENT NUMBERS (v1 → v2)")
    print(f"{'=' * 80}\n")
    print(f"  v1 original failure rate: {v1['failure_rate_orig']}%")
    print(f"  v2 original failure rate: {orig_fr['failure_rate']:.1%}")
    print()
    new_rates = [f"{label}: {model_frs[k]['failure_rate']:.1%}" for k, label in available_models if k in model_frs]
    print(f"  v1 new model range: {v1['failure_rate_range'][0]}% – {v1['failure_rate_range'][1]}%")
    print(f"  v2: {', '.join(new_rates)}")

    # Compare v1 Sonnet signals with v2 Sonnet signals (if available)
    if "sonnet" in model_sigs:
        sonnet_sig_map = {r["signal"]: r for r in model_sigs["sonnet"]}
        print(f"\n  Signal comparison (v1 Sonnet vs v2 Sonnet):")
        print(f"  {'Signal':<30s} {'v1 % Elim':>10s} {'v2 % Elim':>10s} {'Δ':>6s}")
        print(f"  {'-' * 58}")
        for sig, v1s in v1["signals"].items():
            v2r = sonnet_sig_map.get(sig)
            if v2r:
                v2e = f"{v2r['pct_eliminated']:.0f}%" if v2r['pct_eliminated'] is not None else "n/a"
                diff = (v2r['pct_eliminated'] or 0) - v1s['elim']
                print(f"  {sig:<30s} {v1s['elim']:>9}% {v2e:>10s} {diff:>+5.0f}pp")
            else:
                print(f"  {sig:<30s} {v1s['elim']:>9}% {'(n<5)':>10s}")

    print(f"\n  Note: v1 used 1K stratified sample (70/30 problem/clean),")
    print(f"        v2 uses 2K truly random sample.")

    # ═══════════════════════════════════════════════════════════════════════
    # WRITE MARKDOWN
    # ═══════════════════════════════════════════════════════════════════════

    write_markdown(
        orig_fr, available_models, model_frs,
        orig_ad, model_ads,
        model_sigs, orig_turns, model_turns,
        v1,
    )


def write_markdown(
    orig_fr, available_models, model_frs,
    orig_ad, model_ads,
    model_sigs, orig_turns, model_turns,
    v1,
):
    """Write all stats to a markdown file."""
    lines = []
    W = lines.append

    W("# Persistence Experiment v2 — Paper Stats\n")
    W(f"Sample: {orig_fr['n']} conversations (truly random single-turn from score_10k)")
    W(f"Response models: {', '.join(label for _, label in available_models)}\n")

    # ── Failure rates ──
    W("## 1. Failure Rates (Fig failure-persist)\n")
    all_frs = [orig_fr] + [model_frs[k] for k, _ in available_models]
    header = "| Model | N | None | Visible | Invisible | Mixed | Fail % | Invis/Fail |"
    W(header)
    W("|-------|--:|-----:|--------:|----------:|------:|-------:|-----------:|")
    for s in all_frs:
        n = s["n"]
        W(f"| {s['label']} | {n} | {s['none']} ({s['none']/n:.1%}) "
          f"| {s['visible']} ({s['visible']/n:.1%}) "
          f"| {s['invisible']} ({s['invisible']/n:.1%}) "
          f"| {s['mixed']} ({s['mixed']/n:.1%}) "
          f"| {s['failure_rate']:.1%} | {s['invis_of_failures']:.1%} |")

    new_rates = [model_frs[k]['failure_rate'] for k, _ in available_models]
    W("")
    W(f"> Failure rates have gone down from {orig_fr['failure_rate']:.1%} "
      f"to between {min(new_rates):.1%} and {max(new_rates):.1%}.\n")

    # ── Archetype distribution ──
    W("## 2. Archetype Distribution — Invisible+Mixed Failures (Fig arch-persist)\n")

    all_dists = [orig_ad] + [model_ads[k] for k, _ in available_models]
    all_archs = set()
    for d in all_dists:
        all_archs |= set(d["arch_counts"].keys())
    sorted_archs = sorted(all_archs, key=lambda a: -orig_ad["arch_counts"].get(a, 0))

    col_labels = ["GPT-4"] + [label for _, label in available_models]
    W("| Archetype | " + " | ".join(col_labels) + " |")
    W("|-----------|" + "|".join(["------:" for _ in col_labels]) + "|")
    for arch in sorted_archs:
        cells = []
        for d in all_dists:
            c = d["arch_counts"].get(arch, 0)
            pct = c / d["n_invis"] * 100 if d["n_invis"] > 0 else 0
            cells.append(f"{c} ({pct:.1f}%)")
        W(f"| {arch} | {' | '.join(cells)} |")

    W("")
    totals = ", ".join(f"{d['label']} = {d['n_invis']}" for d in all_dists)
    W(f"Totals: {totals} invisible+mixed failures\n")

    # ── Signal tables per model ──
    W("## 3. Problem Signal Deltas (Tab persist-signals)\n")

    def signal_table(results, label, n_orig, n_new):
        W(f"### GPT-4 → {label} (n_orig={n_orig}, n_new={n_new})\n")
        by_elim = sorted(results, key=lambda x: -(x["pct_eliminated"] or -9999))

        W("**Top 5 most reduced (n≥5):**\n")
        W("| Signal | Orig n | Orig % | New n | New % | Δ pp | % Elim |")
        W("|--------|-------:|-------:|------:|------:|-----:|-------:|")
        for r in by_elim[:5]:
            pe = f"{r['pct_eliminated']:.0f}" if r['pct_eliminated'] is not None else "n/a"
            W(f"| {r['signal']} | {r['orig_n']} | {r['orig_pct']:.1f} "
              f"| {r['new_n']} | {r['new_pct']:.1f} | {r['delta_pp']:+.1f} | {pe} |")

        W("")
        W("**Top 5 most persistent (n≥5):**\n")
        W("| Signal | Orig n | Orig % | New n | New % | Δ pp | % Elim |")
        W("|--------|-------:|-------:|------:|------:|-----:|-------:|")
        for r in by_elim[-5:]:
            pe = f"{r['pct_eliminated']:.0f}" if r['pct_eliminated'] is not None else "n/a"
            W(f"| {r['signal']} | {r['orig_n']} | {r['orig_pct']:.1f} "
              f"| {r['new_n']} | {r['new_pct']:.1f} | {r['delta_pp']:+.1f} | {pe} |")
        W("")

    for key, label in available_models:
        if key in model_sigs and key in model_turns:
            signal_table(model_sigs[key], label, len(orig_turns), len(model_turns[key]))

    # ── Cross-model % Eliminated comparison ──
    W("## 4. Cross-Model Comparison: % Eliminated\n")

    sig_maps = {}
    for key, label in available_models:
        if key in model_sigs:
            sig_maps[label] = {r["signal"]: r for r in model_sigs[key]}

    if len(sig_maps) >= 2:
        cmp_labels = list(sig_maps.keys())
        all_sigs = set()
        for sm in sig_maps.values():
            all_sigs |= set(sm.keys())

        W("| Signal | " + " | ".join(cmp_labels) + " | Max gap |")
        W("|--------|" + "|".join(["------:" for _ in cmp_labels]) + "|------:|")

        rows = []
        for sig in all_sigs:
            vals = []
            for label in cmp_labels:
                r = sig_maps[label].get(sig)
                if r and r["pct_eliminated"] is not None:
                    vals.append(r["pct_eliminated"])
                else:
                    vals.append(None)
            non_none = [v for v in vals if v is not None]
            gap = max(non_none) - min(non_none) if len(non_none) >= 2 else 0
            rows.append((sig, vals, gap))

        rows.sort(key=lambda x: -x[2])
        for sig, vals, gap in rows:
            cells = []
            for v in vals:
                cells.append(f"{v:.0f}%" if v is not None else "n/a")
            W(f"| {sig} | {' | '.join(cells)} | {gap:.0f}pp |")

    W("")

    # ── Failure rate ranking ──
    W("## 5. Model Ranking\n")
    ranked = sorted(
        [(label, model_frs[k]["failure_rate"], model_frs[k]["invis_of_failures"])
         for k, label in available_models if k in model_frs],
        key=lambda x: x[1],
    )
    W("| Rank | Model | Failure Rate | Invisible/Fail |")
    W("|-----:|-------|------------:|--------------:|")
    for i, (label, fr, ivf) in enumerate(ranked, 1):
        W(f"| {i} | {label} | {fr:.1%} | {ivf:.1%} |")
    W("")

    # ── v1 comparison ──
    W("## 6. Comparison with Paper's Current Numbers (v1 → v2)\n")
    W(f"- v1 original failure rate: {v1['failure_rate_orig']}%")
    W(f"- v2 original failure rate: {orig_fr['failure_rate']:.1%}")
    v2_rates = ", ".join(f"{label} {model_frs[k]['failure_rate']:.1%}" for k, label in available_models if k in model_frs)
    W(f"- v1 new model range: {v1['failure_rate_range'][0]}% – {v1['failure_rate_range'][1]}%")
    W(f"- v2: {v2_rates}")
    W("")

    if "sonnet" in {k for k, _ in available_models} and "sonnet" in model_sigs:
        sonnet_sig_map = {r["signal"]: r for r in model_sigs["sonnet"]}
        W("| Signal | v1 % Elim | v2 % Elim (Sonnet) | Δ |")
        W("|--------|---------:|---------:|--:|")
        for sig, v1s in v1["signals"].items():
            v2r = sonnet_sig_map.get(sig)
            if v2r:
                v2e = f"{v2r['pct_eliminated']:.0f}%"
                diff = (v2r['pct_eliminated'] or 0) - v1s['elim']
                W(f"| {sig} | {v1s['elim']}% | {v2e} | {diff:+.0f}pp |")
            else:
                W(f"| {sig} | {v1s['elim']}% | (n<5) | — |")
    W("")

    out_path = SCRIPT_DIR / "persistence_v2_paper_stats.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
